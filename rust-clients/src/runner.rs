///
/// runner.rs  Andrew Belles  Dec 2nd, 2025 
///
/// Generic interface for API Clients to complete Jobs  
/// concurrently and safely. 
///
///

use std::sync::Arc; 
use std::time::Duration; 

use async_trait::async_trait; 
use tokio::sync::{mpsc, oneshot}; 
use tokio::task::JoinHandle; 
use tokio::time::timeout;

use crate::config::RunnerConfig; 

#[async_trait]
pub trait Job: Send + Sync + 'static {
    type Output: Send + 'static; 
    type Error: Send + 'static; 

    async fn run(self: Box<Self>) -> Result<Self.Output, Self.Error>; 
    fn name(&self) -> &str; 
}

enum Command<J: Job> {
    Enqueue(J), 
    Shutdown { reply: oneshot::Sender<()> }
}

pub struct Runner<J: Job> {
    transaction: mpsc::Sender<Command<J>>, 
    handles: Vec<JoinHandle<()>>,
    config: RunnerConfig
}

impl<J: Job> Runner<J> {
    pub fn new(config: RunnerConfig) -> Self {
        let (transaction, receipt) = mpsc::channel::<Command<J>>(config.queue_bound); 
        let mut handles = Vec::with_capacity(config.concurrency);

        let receipt = Arc::new(tokio::sync::Mutex::new(receipt));
        for _ in 0..config.concurrency {
            let receipt = Arc::clone(&receipt);
            handles.push(tokio::spawn(async move {
                loop {
                    let command = {
                        let mut guard = receipt.lock().await; 
                        guard.recv().await 
                    };
                    match command {
                        Some(Command::Enqueue(job)) => {
                            let _ = job.run().await; 
                        }
                        Some(Command::Shutdown { reply }) => {
                            let _ = reply.send(()); 
                            break; 
                        }
                        None => break, 
                    }
                }
            }))
        }  

        Runner { transaction, handles, config }
    }
}
