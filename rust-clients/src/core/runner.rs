///
/// runner.rs  Andrew Belles  Dec 2nd, 2025 
///
/// Generic interface for API Clients to complete Jobs  
/// concurrently and safely. 
/// 
/// Work is pushed into bounded mpsc queue and distributed to 
/// concurrency (#) worker jobs 
///
/// Shared state block tracks queued/active/processed counts 
///
/// Shutdown is graceful/cooperative 
///

use std::sync::{
    Arc, 
    atomic::{AtomicBool, AtomicUsize, Ordering}
}; 

use std::time::Duration; 

use async_trait::async_trait; 
use tokio::sync::{mpsc, oneshot, Mutex}; 
use tokio::task::JoinHandle; 
use tokio::time::timeout;

use crate::core::config::RunnerConfig; 

#[async_trait]
pub trait Job: Send + Sync + 'static {
    type Output: Send + 'static; 
    type Error: Send + 'static; 

    async fn run(self: Box<Self>) -> Result<Self::Output, Self::Error>; 
    fn name(&self) -> &str; 
}

enum Command<J: Job> {
    Enqueue(J), 
    Shutdown { reply: oneshot::Sender<()> }
}

pub struct Snapshot {
    pub queued: usize, 
    pub processed: usize, 
    pub running: bool, 
    pub last_activity: std::time::Instant 
}

struct AtomicSnapshot {
    queued: AtomicUsize, 
    processed: AtomicUsize, 
    active: AtomicUsize, 
    running: AtomicBool, 
    last_activity: tokio::sync::RwLock<std::time::Instant> 
}

pub struct Runner<J: Job> {
    transaction: mpsc::Sender<Command<J>>, 
    receipt: Arc<Mutex<mpsc::Receiver<Command<J>>>>,
    handles: Vec<JoinHandle<()>>,
    idle_handle: Option<JoinHandle<()>>, 
    state: Arc<AtomicSnapshot>, 
    config: RunnerConfig,

    on_idle: Option<Arc<dyn Fn() + Send + Sync>>, 
    on_error: Option<Arc<dyn Fn(&J::Error) + Send + Sync>> 
}

impl<J: Job> Runner<J> {
    pub fn new(config: RunnerConfig) -> Self {
        let (transaction, receipt) = mpsc::channel::<Command<J>>(config.queue_bound); 
        Self {
            transaction, 
            receipt: Arc::new(Mutex::new(receipt)), 
            handles: Vec::with_capacity(config.concurrency), 
            idle_handle: None, 
            state: Arc::new(AtomicSnapshot::new()),
            config, 
            on_idle: None, 
            on_error: None
        }
    } 

    pub fn set_on_idle<F>(&mut self, f: F)
    where 
        F: Fn() + Send + Sync + 'static
    {
        self.on_idle = Some(Arc::new(f))
    }

    pub fn set_on_error<F>(&mut self, f: F)
    where 
        F: Fn(&J::Error) + Send + Sync + 'static 
    {
        self.on_error = Some(Arc::new(f))
    }

    pub fn start(&mut self) {
        // Only first call to start subworkers through Mutex on running 
        if self.state.running.swap(true, Ordering::Relaxed) {
            return; // guard against multiple starts 
        }

        for _ in 0..self.config.concurrency {
            // Get single mpsc receiver behind Mutex, shared state, and error handler  
            let receipt  = Arc::clone(&self.receipt); 
            let state    = Arc::clone(&self.state);
            let on_error = self.on_error.clone(); 

            self.handles.push(tokio::spawn(async move {
                loop {
                    // acquire lock 
                    let command = {
                        let mut guard = receipt.lock().await; 
                        guard.recv().await
                    };

                    match command {
                        // Try to complete the job
                        Some(Command::Enqueue(job)) => {
                            state.queued.fetch_sub(1, Ordering::Relaxed);
                            state.active.fetch_add(1, Ordering::Relaxed);
                            
                            // Run job, await completion, then adjust state atomically 
                            let result = Box::new(job).run().await; 
                            state.active.fetch_sub(1, Ordering::Relaxed);
                            state.processed.fetch_add(1, Ordering::Relaxed);
                            state.touch().await; 
                            if let Err(err) = result {
                                if let Some(callback) = &on_error {
                                    callback(&err)
                                }
                            }
                        }
                        // Shutdown requested during call
                        Some(Command::Shutdown { reply }) => {
                            let _ = reply.send(()); 
                            break; 
                        }
                        None => break, 
                    }
                }
            }));
        }

        // Shutdown runner after completion of job 
        if let Some(idle_secs) = self.config.idle_shutdown_secs {
            // Single transaction from mpsc, idle handler, timings 
            let state = Arc::clone(&self.state); 
            let transaction = self.transaction.clone(); 
            let on_idle = self.on_idle.clone(); 
            let graceful = self.config.graceful_shutdown_secs; 

            // Respects graceful shutdown while checking for idle runner  
            self.idle_handle = Some(tokio::spawn(async move {
                let idle_for = Duration::from_secs(idle_secs);
                loop {
                    // Wait half of idle time 
                    tokio::time::sleep(idle_for / 2).await; 
                    if state.is_idle_for(idle_for).await {
                        if let Some(callback) = &on_idle {
                            callback(); 
                        }

                        // Respect graceful shutdown 
                        for _ in 0..graceful.max(1) {
                            let (s, r) = oneshot::channel(); 
                            if transaction.send(Command::Shutdown { reply: s }).await.is_err() {
                                break;
                            }
                            let _ = timeout(Duration::from_secs(graceful), r).await; 
                        }
                        break; 
                    }
                }
            }));
        }
    }

    pub fn enqueue(&self, job: J) -> bool {
        if self.transaction.try_send(Command::Enqueue(job)).is_ok() {
            self.state.queued.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false 
        }
    }

    pub async fn request_stop(&self) {
        for _ in 0..self.config.concurrency {
            let (s, r) = oneshot::channel(); 
            if self.transaction.send(Command::Shutdown { reply: s }).await.is_err() {
                break;
            } 
            let _ = timeout(
                Duration::from_secs(self.config.graceful_shutdown_secs),
                r,
            ).await; 
        }
    }

    pub async fn join(&mut self) {
        if let Some(handle) = self.idle_handle.take() {
            let _ = handle.await; 
        }
        for handle in self.handles.drain(..) {
            let _ = handle.await; 
        }
        self.state.running.store(false, Ordering::Relaxed);
    }

    pub fn backlog(&self) -> usize {
        self.state.queued.load(Ordering::Relaxed)
    }

    pub async fn snapshot(&self) -> Snapshot {
        self.state.snapshot().await 
    }
}

impl AtomicSnapshot {
    fn new() -> Self {
        let now = std::time::Instant::now(); 
        Self {
            queued: AtomicUsize::new(0),
            processed: AtomicUsize::new(0), 
            active: AtomicUsize::new(0),
            running: AtomicBool::new(false),
            last_activity: tokio::sync::RwLock::new(now)
        }
    }

    async fn touch(&self) {
        let mut guard = self.last_activity.write().await; 
        *guard = std::time::Instant::now(); 
    }

    async fn snapshot(&self) -> Snapshot {
        Snapshot {
            queued: self.queued.load(Ordering::Relaxed),
            processed: self.processed.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
            last_activity: *self.last_activity.read().await
        }
    }

    async fn is_idle_for(&self, dur: Duration) -> bool {
        let last = *self.last_activity.read().await; 
        self.queued.load(Ordering::Relaxed) == 0 && 
            self.active.load(Ordering::Relaxed) == 0 &&
            last.elapsed() >= dur 
    }
}
