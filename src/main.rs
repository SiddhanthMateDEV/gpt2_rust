// For tensorflow
extern crate tensorflow;
use tensorflow::{Graph,Session};
// For time clocks
use std::time::Instant;
// for datasets
extern crate ndarray;
use ndarray::Array;
// For os-level interactions 
use std::fs;
use std::env;
// Part-1-Imports
extern crate clap;
use std::fs;
use std::io;
//  Part-2-Imports
use clap::{App, Arg};

// --------------------------------------------------------------------------------------------------------------------
// storing the last saved training checkpoint for
const CHECKPOINT_DIR: &str= "checkpoint";
const SAMPLE_DIR: &str= "samples";

// Part-1----------------------------------------------------------------
// making directory for all storing all the data training checkpoints 
fn make_checkpoint_Directory(path: &str)-> std::io::Result<()> {
    // Creating directories with error handling 
    fs::create_dir_all(path).map_err(|e| io::Error::new(io::Error::new(e.kind(),format!("Error (Part-1) creating directory for path{}:{}",path,e))));
}

// Part-2----------------------------------------------------------------
pub fn parser()-> Result<clap::ArgMatches<'static'>,clap::Error>{
    // Argument matching for the command line to enter training attributes or set defaults
    let matches=App::new("Fine-tune GPT-2 on your custom dataset")
                    .version("1.0")
                    .author("SiddhanthParagMate")
                    .about("GPT-2 Code with multi-core optimization")
                    .arg(Arg::with_name("dataset")
                        .help("Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files)")
                        .required(true))
                    .arg(Arg::with_name("model-name")
                        .help("Pretrained model name")
                        .default_value("117M"))
                    .arg(Arg::with_name("combine")
                        .help("Concatenate input files with separator into chunks of this minimum size")      
                        .default("50000"))
                    .arg(Arg::with_name("batch_size")
                        .help("batch size")
                        .default("1"))
                    .arg(Arg::with_name("learning_rate")
                        .help("learning rate for Adam")
                        .default_value("0.0001"))
                    .arg(Arg::with_name("accumulate_gradients")
                        .help("Accumulate gradients across N minibatches.")
                        .default_value("1"))
                    .arg(Arg::with_name("memory_saving_gradients")
                        .help("Use gradient checkpointing to reduce vram usage.")
                        .takes_value(false))
                    .arg(Arg::with_name("only_train_transformer_layers")
                        .help("Restrict training to the transformer blocks.")
                        .takes_value(false))
                    .arg(Arg::with_name("restore_from")
                        .help("Either \"latest\", \"fresh\", or a path to a checkpoint file")
                        .default_value("latest"))
                    .arg(Arg::with_name("run_name")
                        .help("Run id. Name of subdirectory in checkpoint/ and samples/")
                        .default_value("run1"))
                    .arg(Arg::with_name("sample_every")
                        .help("Generate samples every N steps")
                        .default_value("100"))
                    .arg(Arg::with_name("sample_length")
                        .help("Sample this many tokens")
                        .default_value("1023"))
                    .arg(Arg::with_name("sample_num")
                        .help("Generate this many samples")
                        .default_value("1"))
                    .arg(Arg::with_name("save_every")
                        .help("Write a checkpoint every N steps")
                        .default_value("1000"))
                    .arg(Arg::with_name("val_dataset")
                        .help("Dataset for validation loss, defaults to --dataset.")
                        .default_value("None"))
                    .arg(Arg::with_name("val_batch_size")
                        .help("Batch size for validation.")
                        .default_value("2"))
                    .arg(Arg::with_name("val_batch_count")
                        .help("Number of batches for validation.")
                        .default_value("40"))
                    .arg(Arg::with_name("val_every")
                        .help("Calculate validation loss every STEPS steps.")
                        .default_value("0"))
                    .arg(Arg::with_name("stop_after")
                        .help("Stop after training counter reaches STOP")
                        .default_value("None"))
                    .get_matches();
}

// struct TrainingGpt{
//     args: Args,
//     enc:Encoders,
//     hparams: Hyperparameters,
//     config: TensorFlowConfiguration,
//     session: TensorFlowSession,
// }



// impl  TrainingGpt{
//     pub fn argsParser(&self){
//         match parser(){
//             Ok(Matches)=>{
//                 // Pull the dataset value from the command line
//                 if let Some(dataset) = matches.value_of("dataset") {
//                     println!("The path to the dataset is : {}", dataset)
//                 }
    
//                 let model_name=matches.value_of("model_name").unwrap_or("117M");
//                 println!("The name of the model is : {}", model_name);

//                 // Combine datasets into one dataset or set the token default to 500000
//                 let combine=matches.value_of("combine_arg")
//                                       .unwrap_or("50000")
//                                       .parse::<i32>()
//                                       .unwrap_or(50000);
                
//                 // add the rest of them here 
    
//             }
//         }

//     // priv fn(&self)-> {

//     // }
    
//     }
// }

struct HyperParameters{
    n_ctx: usize,
    sample_length: usize,
}

impl HyperParameters {
    fn load_from_file()
}


fn main() {

}
