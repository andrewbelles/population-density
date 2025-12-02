mod config; 
mod http; 

fn main() {
    let cfg = config::AppConfig::default(); 

    println!("{:?}", cfg);
    println!("Hello, world!");
}
