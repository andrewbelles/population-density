///
/// pagination.rs  Andrew Belles  Dec 2nd, 2025 
///
/// 
///
///
///

use crate::config::{PaginationConfig, PaginationFSM};

#[derive(Debug, Clone)]
pub struct PaginationState {
    pub pages_fetched: u32, 
    pub last_cursor: Option<String>, 
    pub last_batch_len: usize
}

impl PaginationState {
    pub fn new() -> Self {
        Self {
            pages_fetched: 0,
            last_cursor: None, 
            last_batch_len: 0
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pager {
    config: PaginationConfig, 
    state: PaginationState
}

impl Pager {
    pub fn new(config: PaginationConfig) -> Self {
        Pager { config, state: PaginationState::new() }
    }

    // Determine parameters to inject to json request using current state of FSM 
    pub fn next_params(&self) -> Option<Vec<(String, String)>> {
        if let Some(max) = self.config.max_pages {
            if self.state.pages_fetched >= max {
                return None;
            }
        }

        let mut params = Vec::with_capacity(3);
        let names = &self.config.param_names;

        match self.config.strategy {
            PaginationFSM::OffsetLimit => {
                let limit = self.config.page_size.unwrap_or(100);
                params.push((names.offset.clone(), 
                    (self.state.pages_fetched as u64 * limit as u64).to_string()));
                params.push((names.per_page.clone(), limit.to_string()));
            }

            PaginationFSM::PageNumber => {
                let page = self.state.pages_fetched + 1; 
                params.push((names.page.clone(), page.to_string())); 
                if let Some(size) = self.config.page_size {
                    params.push((names.per_page.clone(), size.to_string())); 
                }
            }

            PaginationFSM::Cursor => {
                if let Some(cursor) = self.state.last_cursor.as_ref()
                    .or(self.config.initial_cursor.as_ref()) {
                    params.push((names.cursor.clone(), cursor.clone()));
                }

                if let Some(size) = self.config.page_size {
                    params.push((names.per_page.clone(), size.to_string()));
                }
            }
        }

        Some(params)
    }

    // Logic for whether we can advance the state of the FSM
    pub fn advance_state(&mut self, next_cursor: Option<String>, 
        has_more: Option<bool>, batch_len: usize) -> bool {
        self.state.pages_fetched += 1; 
        self.state.last_batch_len = batch_len; 

        if let Some(cursor) = next_cursor {
            if (self.config.stop_on_duplicate && 
                self.state.last_cursor.as_ref() == Some(&cursor)) {
                return false; 
            }
            self.state.last_cursor = Some(cursor); 
        }

        if let Some(has_more) = has_more {
            return has_more && !self.reached_page_cap();
        }

        if batch_len == 0 {
            return false;
        }

        if matches!(self.config.strategy, 
            PaginationFSM::OffsetLimit | PaginationFSM::PageNumber) {
            if let Some(size) = self.config.page_size {
                if batch_len < size as usize {
                    return false; 
                }
            }
        }

        !self.reached_page_cap() 
    }

    fn reached_page_cap(&self) -> bool {
        matches!(self.config.max_pages, Some(max) if self.state.pages_fetched >= max)
    }
}
