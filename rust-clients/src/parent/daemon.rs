///
/// daemon.rs  Andrew Belles  Dec 3rd, 2025 
///
/// Parent Daemon that Clients register with. 
/// Wait-free, CPU-Scheduler-like design which treats 
/// rate limits as IO bursts 
///

use std::{
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    time::{Duration, Instant}
}; 

use tokio::{sync::{mpsc, Notify}, task::JoinHandle}; 
use tracing::warn; 

use crate::core::runner::{Job, Runner}; 

static MS: f64 = 1e-6; 

/************ Main Interface with Daemon ******************/ 

/************ ClientSpec **********************************/ 
/* All information required for a Client to register with 
 * the Daemon. 
 */ 
pub struct ClientSpec {
    pub id: ClientId, 
    pub deps: Vec<ClientId>, 
    pub max_tokens: u32, 
    pub refill_per_minute: u32 
}

/************ Spawn Daemon ********************************/ 
/* Given a list of Client Specifications, a single Runner 
* and a DAG of dependencies, generates a Daemon to handle 
* requests etc. 
*
* Caller Provides: 
*   Valid Runner to handle threading/concurrency
*   Array of ClientSpec to register with Daemon 
*   Edges of DAG outlining dependencies 
*
* Note:
*   Clients cannot be registered with daemon after spawning it 
*/ 
pub fn spawn_daemon<J: Job + Copy + Send + Sync + 'static>(
    runner: Runner<J>, clients: &[ClientSpec], edges: &[(ClientId, ClientId)]
) -> (mpsc::Sender<DaemonEvent<J>>, JoinHandle<()>) {
    let (event_transaction, event_receipt) = mpsc::channel::<DaemonEvent<J>>(1024); 
    let mut daemon = Daemon::new(runner); 

    for client in clients {
        daemon.register_client(
            client.id.clone(), 
            client.deps.clone(), 
            client.max_tokens, 
            client.refill_per_minute
        );
    }

    for (up, down) in edges {
        daemon.add_dependency(up, down);
    }

    let handle = tokio::spawn(async move {
        daemon.run(event_receipt).await; 
    });

    (event_transaction, handle)
}


/************ Bindings ************************************/ 
pub type ClientId = String;  // Unique Id given to Client 
pub type JobId    = String;  // Unique Id given to a Job for a Client 

/************ RateLimiter *********************************/
/* Attaches to a Client to denote how many requests are alloted to them  
 */ 
#[derive(Debug)]
struct RateLimiter {
    tokens: f64, 
    max_tokens: f64, 
    refill_per_sec :f64, 
    last_refill: Instant 
}

impl RateLimiter {
    fn new(max_tokens: u32, refill_per_minute: u32) -> Self {
        let now = Instant::now(); 
        Self {
            tokens: max_tokens as f64, 
            max_tokens: max_tokens as f64, 
            refill_per_sec: refill_per_minute as f64 / 60.0_f64, 
            last_refill: now 
        }
    }

    // Try and acquire a single job from a client via RateLimiter 
    fn try_acquire(&mut self) -> bool {
        self.refill(); 
        if self.tokens >= 1.0 {
            self.tokens -= 1.0; 
            true 
        } else {
            false 
        }
    }

    // Determine time when Client can next make a request 
    fn next_ready(&mut self) -> Instant {
        self.refill(); 
        if self.tokens >= 1.0 {
            Instant::now() 
        } else {
            let deficit = 1.0 - self.tokens; 
            let secs = deficit / self.refill_per_sec.max(MS); 
            Instant::now() + Duration::from_secs_f64(secs)
        }
    }

    // Perform a refill on rate tokens for client 
    fn refill(&mut self) {
        let now = Instant::now(); 
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            self.tokens = (self.tokens + elapsed * self.refill_per_sec).min(self.max_tokens);
            self.last_refill = now; 
        }
    }
}

/************ ClientState *********************************/ 
/* Determines the working state of a client. 
 * 
 * If a client has dependencies, and how many of them are yet 
 * to complete their active jobs, as well as the Client's RateLimiter,
 * and jobs that need to be injected 
 * 
 */
#[derive(Debug)]
struct ClientState<J: Job> {
    dependencies_remaining: HashSet<ClientId>, 
    dependencies: Vec<ClientId>, 
    ready: bool, 
    seeds: VecDeque<JobPayload<J>>, 
    limiter: RateLimiter, 
    inflight: usize 
}

/************ JobPayload **********************************/ 
/* A single job defined by its Id, the ClientId its attached to 
 * and the job itself 
 */
#[derive(Debug, Clone)]
pub struct JobPayload<J: Job> {
    pub id: JobId, 
    pub client: ClientId, 
    pub job: J
}

/************ Deferred ************************************/ 
/* a single job that has been deferred. Includes 
 * the job and time of deferrment bin 
 */  
#[derive(Debug)]
struct Deferred<J: Job> {
    when: Instant, 
    payload: JobPayload<J> 
}

/************ Ordering for Deferred ***********************/ 

// Ordering for Deferred required for min-heap/BinaryHeap
impl<J: Job> Ord for Deferred<J> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.when.cmp(&other.when).reverse() // reverse for min-heap 
    }
}

impl<J: Job> PartialOrd for Deferred<J> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<J: Job> PartialEq for Deferred<J> {
    fn eq(&self, other: &Self) -> bool {
        self.when == other.when 
    }
}

impl<J: Job> Eq for Deferred<J> {} 

/************ DaemonEvent *********************************/ 
pub enum DaemonEvent<J: Job> {
    NewJob(JobPayload<J>), // New job recieved for a client 
    ClientReady(ClientId), // Client can start a job 
    JobFinished(JobId)     // Release JobId, job is completed 
}

/************ Daemon **************************************/ 
/* Wait-Free daemon, respecting rate limits and dependencies  
 * of registered clients through CPU-Scheduler algorithm and 
 * DAG dependency graph 
 *
 * J implements Job outlined in core/job.rs. 
 */ 
pub struct Daemon<J: Job> {
    clients: HashMap<ClientId, ClientState<J>>, 
    running_ids: HashSet<JobId>, 
    ready_transaction: mpsc::Sender<JobPayload<J>>, 
    ready_receipt: mpsc::Receiver<JobPayload<J>>, 
    deferred: BinaryHeap<Deferred<J>>, 
    notify: Notify, 
    runner: Runner<J> 
}

impl<J: Job + Copy> Daemon<J> {
    pub fn new(runner: Runner<J>) -> Self {
        let (ready_transaction, ready_receipt) = mpsc::channel(1024); 
        Self {
            clients: HashMap::new(), 
            running_ids: HashSet::new(), 
            ready_transaction, 
            ready_receipt, 
            deferred: BinaryHeap::new(), 
            notify: Notify::new(), 
            runner 
        }
    }

    // Registers a single client into the Daemon 
    pub fn register_client(
        &mut self, id: ClientId, deps: Vec<ClientId>, 
        max_tokens: u32, refill_per_minute: u32) {
        let state = ClientState {
            dependencies_remaining: deps.iter().cloned().collect(), 
            dependencies: deps.clone(), 
            ready: deps.is_empty(), 
            seeds: VecDeque::new(), 
            limiter: RateLimiter::new(max_tokens, refill_per_minute), 
            inflight: 0 
        };
        self.clients.insert(id, state); 
    }

    // Adds a single dependency for a Client 
    pub fn add_dependency(&mut self, upstream: &ClientId, downstream: &ClientId) {
        if let Some(state) = self.clients.get_mut(upstream) {
            state.dependencies.push(downstream.clone()); 
        }
    }

    // Performs actions on clients registered with Daemon per event 
    pub async fn run(mut self, mut event_receipt: mpsc::Receiver<DaemonEvent<J>>) {
        self.runner.start(); 

        loop {
            tokio::select! {
                Some(event) = event_receipt.recv() => {
                    self.handle_event(event).await; 
                },
                Some(payload) = self.ready_receipt.recv() => {
                    let JobPayload { id, client, job } = payload; 
                    if self.runner.enqueue(job) {
                        self.running_ids.insert(id);
                    } else {
                        let _ = self.ready_transaction.try_send(JobPayload{ id, client, job }); 
                    }
                },
                else => break,
            }

            self.drain_deferred().await; 

        }

        self.runner.join().await; 
    }

    // Event handler for a single DaemonEvent 
    async fn handle_event(&mut self, event: DaemonEvent<J>) {
        match event {
            DaemonEvent::NewJob(payload) => {
                // Avoid duplicate jobs 
                if self.running_ids.contains(&payload.id) {
                    return; 
                }

                // Get ref to client payload references 
                if let Some(state) = self.clients.get_mut(&payload.client) {
                    // Client is not ready for job execution, push into deque 
                    if !state.ready {
                        state.seeds.push_back(payload); 
                        return; 
                    }

                    send_job_or_defer(&self.ready_transaction, &mut self.deferred, 
                        state, payload);
                } else {
                    warn!("Unknown client {}", payload.client);
                }
            }
            // Client is ready for action, see if it has any jobs in its personal dequeue 
            DaemonEvent::ClientReady(client_id) => {
                if let Some(state) = self.clients.get_mut(&client_id) {
                    state.ready = true ;
                    while let Some(payload) = state.seeds.pop_front() {
                        send_job_or_defer(&self.ready_transaction, 
                            &mut self.deferred, state, payload);
                    }
                }

                if let Some(state) = self.clients.get(&client_id) {
                    let deps: Vec<_> = state.dependencies.clone(); 
                    let _ = state; 

                    for dep in deps {
                        if let Some(down) = self.clients.get_mut(&dep) {
                            down.dependencies_remaining.remove(&client_id); 
                            if down.dependencies_remaining.is_empty() {
                                down.ready = true; 
                                while let Some(payload) = down.seeds.pop_front() {
                                    send_job_or_defer(&self.ready_transaction, 
                                        &mut self.deferred, down, payload);
                                }
                            }
                        }
                    }
                }
                self.notify.notify_one(); 
            }

            // Remove job id from tracked 
            DaemonEvent::JobFinished(job_id) => {
                self.running_ids.remove(&job_id);
            }
        }
    }

    // Try and drain deferred jobs from the deferred min-heap 
    async fn drain_deferred(&mut self) {
        let now = Instant::now(); 
        while let Some(peek) = self.deferred.peek() {
            // Not yet ready to drain off deferred heap 
            if peek.when > now {
                break; 
            } 

            if let Some(Deferred { payload, .. }) = self.deferred.pop() { 
                if let Some(state) = self.clients.get_mut(&payload.client) {
                    send_job_or_defer(&self.ready_transaction, &mut self.deferred, state, payload);
                }
            }
        }
    }
} 

async fn send_job_or_defer<J: Job>(
    ready_transaction: &mpsc::Sender<JobPayload<J>>, 
    deferred: &mut BinaryHeap<Deferred<J>>, 
    state: &mut ClientState<J>, 
    payload: JobPayload<J>) 
{
    if state.limiter.try_acquire() {
        let _ = ready_transaction.try_send(payload); 
    } else {
        let when = state.limiter.next_ready(); 
        deferred.push(Deferred { when, payload })
    }
}
