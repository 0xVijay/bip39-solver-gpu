pub mod config;
pub mod word_space;
pub mod eth;
pub mod slack;
pub mod job_types;
pub mod job_server;
pub mod worker_client;
pub mod gpu_backend;
pub mod opencl_backend;
pub mod cuda_backend;
pub mod gpu_manager;

#[cfg(test)]
mod tests;