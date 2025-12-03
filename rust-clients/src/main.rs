mod http; 
mod config; 
mod pagination;
mod crawler; 

fn main() {
    let hc = http::HttpConfig::new();
    let cfg = config::AppConfig::default(); 

    println!("{:?}", cfg);
    println!("Hello, world!");
}
