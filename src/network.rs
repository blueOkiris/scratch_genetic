/*
 * Author: Dylan Turner
 * Description: Neural Network that can be used to predict a game
 */

use std::{
    fs::{
        File, remove_file
    }, io::{
        Write, Read
    }, path::Path
};
use futures::future::try_join_all;
use tokio::{
    spawn,
    task::JoinHandle
};
use crate::neuron::{
    NeuronConnectionMap, NeuronConnection, NeuronConnectionSet
};

#[derive(Debug, Clone)]
pub struct Network {
    pub maps: Vec<NeuronConnectionMap>,

    // Tweakable settings
    pub layer_sizes: Vec<usize>,
    pub num_inputs: usize,
    pub num_outputs: usize
}

const NUM_USIZE_BYTES: usize = (usize::BITS / 8) as usize;

impl Network {
    /*
     * Generate all connections randomly
     * Unlike underlying functions these ARE faster when multithreaded
     */
    pub async fn new_random(
            layer_sizes: Vec<usize>, num_inputs: usize, num_outputs: usize,
            activation_thresh: f64, trait_swap_chance: f64,
            weight_mutate_chance: f64, weight_mutate_amount: f64,
            offset_mutate_chance: f64, offset_mutate_amount: f64) -> Self {
        let mut handles: Vec<JoinHandle<NeuronConnectionMap>> = Vec::new();
        for i in 0..=layer_sizes.len() {
            handles.push(spawn(
                if i == 0 {
                    NeuronConnectionMap::new_random(
                        layer_sizes[i], num_inputs,
                        activation_thresh, trait_swap_chance,
                        weight_mutate_chance, weight_mutate_amount,
                        offset_mutate_chance, offset_mutate_amount
                    )
                } else if i == layer_sizes.len() {
                    NeuronConnectionMap::new_random(
                        num_outputs, layer_sizes[i - 1],
                        activation_thresh, trait_swap_chance,
                        weight_mutate_chance, weight_mutate_amount,
                        offset_mutate_chance, offset_mutate_amount
                    )
                } else {
                    NeuronConnectionMap::new_random(
                        layer_sizes[i], layer_sizes[i - 1],
                        activation_thresh, trait_swap_chance,
                        weight_mutate_chance, weight_mutate_amount,
                        offset_mutate_chance, offset_mutate_amount
                    )
                }
            ));
        }
        Self {
            maps: try_join_all(handles).await.unwrap(),
            layer_sizes,
            num_inputs,
            num_outputs
        }
    }

    // Cannot be parallelized.
    pub async fn result(&self, input_bits: &Vec<u8>) -> Vec<u8> {
        let mut last_bits = input_bits.clone();
        for map in self.maps.iter() {
            last_bits = map.layer_activations(&last_bits).await;
        }
        last_bits.clone()
    }

    // Can't be parallelized bc mutation
    pub async fn random_trade(&mut self, other: &mut Self) {
        for i in 0..self.maps.len() {
            self.maps[i].trade_with(&mut other.maps[i]).await;
        }
    }

    // Can't be parallelized bc mutation
    pub async fn mutate(&mut self) {
        for map in self.maps.iter_mut() {
            map.mutate_all().await;
        }
    }

    // Don't care to optimize. Performance doesn't really matter
    pub fn from_file(fname: &str) -> Self {
        let mut file = File::open(fname).expect("Failed to open model file!");

        // Get layer data
        let mut num_inputs_data: [u8; NUM_USIZE_BYTES] = [0; NUM_USIZE_BYTES];
        file.read_exact(&mut num_inputs_data).expect("Failed to load model from file!");
        let num_inputs = usize::from_be_bytes(num_inputs_data);
        let mut num_outputs_data: [u8; NUM_USIZE_BYTES] = [0; NUM_USIZE_BYTES];
        file.read_exact(&mut num_outputs_data).expect("Failed to load model from file!");
        let num_outputs = usize::from_be_bytes(num_outputs_data);
        let mut layer_sizes = Vec::new();
        let mut layer_size_data: [u8; NUM_USIZE_BYTES] = [0; NUM_USIZE_BYTES];
        file.read_exact(&mut layer_size_data).expect("Failed to load model from file!");
        while usize::from_be_bytes(layer_size_data) != 0xFFFFFFFFFFFFFFFF {
            layer_sizes.push(usize::from_be_bytes(layer_size_data));
            file.read_exact(&mut layer_size_data).expect("Failed to load model from file!");
        }

        // Construct a big array from the rest
        let mut big_arr_size = num_inputs * layer_sizes[0];
        for i in 0..layer_sizes.len() - 1 {
            big_arr_size += layer_sizes[i] * layer_sizes[i + 1];
        }
        big_arr_size += layer_sizes[layer_sizes.len() - 1] * num_outputs;
        big_arr_size *= 8 * 6; // 8 for wgt, ofst, wgt & ofst chance & amnt
        for i in 0 as usize..=layer_sizes.len() {
            let out_layer_size = if i == layer_sizes.len() {
                num_outputs
            } else {
                layer_sizes[i]
            };
            big_arr_size += 2 * 8 * out_layer_size;
        }
        let mut big_arr = vec![0; big_arr_size];

        file.read_exact(&mut big_arr).expect("Failed to load model from file!");

        let mut x = 0;
        let mut maps = Vec::new();
        for i in 0 as usize..=layer_sizes.len() {
            let in_layer_size = if i == 0 {
                num_inputs
            } else {
                layer_sizes[i - 1]
            };
            let out_layer_size = if i == layer_sizes.len() {
                num_outputs
            } else {
                layer_sizes[i]
            };

            let mut map = Vec::new();
            for _ in 0..out_layer_size {
                let mut conns = Vec::new();
                for _ in 0..in_layer_size {
                    let mut weight_data = [0; 8];
                    for k in 0..8 {
                        weight_data[k] = big_arr[x];
                        x += 1;
                    }
                    let mut offset_data = [0; 8];
                    for k in 0..8 {
                        offset_data[k] = big_arr[x];
                        x += 1;
                    }
                    let mut weight_mutate_chance_data = [0; 8];
                    for k in 0..8 {
                        weight_mutate_chance_data[k] = big_arr[x];
                        x += 1;
                    }
                    let mut weight_mutate_amount_data = [0; 8];
                    for k in 0..8 {
                        weight_mutate_amount_data[k] = big_arr[x];
                        x += 1;
                    }
                    let mut offset_mutate_chance_data = [0; 8];
                    for k in 0..8 {
                        offset_mutate_chance_data[k] = big_arr[x];
                        x += 1;
                    }
                    let mut offset_mutate_amount_data = [0; 8];
                    for k in 0..8 {
                        offset_mutate_amount_data[k] = big_arr[x];
                        x += 1;
                    }
                    conns.push(
                        NeuronConnection {
                            weight: f64::from_be_bytes(weight_data),
                            offset: f64::from_be_bytes(offset_data),
                            weight_mutate_chance: f64::from_be_bytes(weight_mutate_chance_data),
                            weight_mutate_amount: f64::from_be_bytes(weight_mutate_amount_data),
                            offset_mutate_chance: f64::from_be_bytes(offset_mutate_chance_data),
                            offset_mutate_amount: f64::from_be_bytes(offset_mutate_amount_data)
                        }
                    );
                }
                let mut activation_thresh_data = [0; 8];
                for k in 0..8 {
                    activation_thresh_data[k] = big_arr[x];
                    x += 1;
                }
                let mut trait_swap_chance_data = [0; 8];
                for k in 0..8 {
                    trait_swap_chance_data[k] = big_arr[x];
                    x += 1;
                }
                map.push(NeuronConnectionSet {
                    conns,
                    activation_thresh: f64::from_be_bytes(activation_thresh_data),
                    trait_swap_chance: f64::from_be_bytes(trait_swap_chance_data),
                });
            }

            maps.push(NeuronConnectionMap {
                map
            });
        }

        Self {
            maps,
            layer_sizes,
            num_inputs,
            num_outputs
        }
    }

    // Don't care to optimize. Performance doesn't really matter
    pub async fn save_model(&self, fname: &str) {
        if Path::new(fname).exists() {
            remove_file(fname).unwrap();
        }

        let mut file = File::create(fname).expect("Failed to open model file!");

        // Write a header with the correct layer size stuff
        file.write_all(&self.num_inputs.to_be_bytes()).expect("Failed to save model file!");
        file.write_all(&self.num_outputs.to_be_bytes()).expect("Failed to save model file!");
        for layer_size in self.layer_sizes.iter() {
            file.write_all(&layer_size.to_be_bytes()).expect("Failed to save model file!");
        }
        file.write_all(&(0xFFFFFFFFFFFFFFFF as usize).to_be_bytes())
            .expect("Failed to save model file!");

        // Then construct an array with the correct sizes
        let mut big_arr_size = self.num_inputs * self.layer_sizes[0];
        for i in 0..self.layer_sizes.len() - 1 {
            big_arr_size += self.layer_sizes[i] * self.layer_sizes[i + 1];
        }
        big_arr_size += self.layer_sizes[self.layer_sizes.len() - 1] * self.num_outputs;
        big_arr_size *= 8 * 6; // 8 for weight, offset, weight & offset mutate chance & amount
        for i in 0 as usize..=self.layer_sizes.len() {
            let out_layer_size = if i == self.layer_sizes.len() {
                self.num_outputs
            } else {
                self.layer_sizes[i]
            };
            big_arr_size += 2 * 8 * out_layer_size;
        }
        let mut big_arr = vec![0; big_arr_size];

        let mut x = 0;
        for map in self.maps.iter() {
            for conns in map.map.iter() {
                for conn in conns.conns.iter() {
                    let weight_data = conn.weight.to_be_bytes();
                    for k in 0..8 {
                        big_arr[x] = weight_data[k];
                        x += 1;
                    }
                    let offset_data = conn.offset.to_be_bytes();
                    for k in 0..8 {
                        big_arr[x] = offset_data[k];
                        x += 1;
                    }
                    let weight_mutate_chance_data = conn.weight_mutate_chance.to_be_bytes();
                    for k in 0..8 {
                        big_arr[x] = weight_mutate_chance_data[k];
                        x += 1;
                    }
                    let weight_mutate_amount_data = conn.weight_mutate_amount.to_be_bytes();
                    for k in 0..8 {
                        big_arr[x] = weight_mutate_amount_data[k];
                        x += 1;
                    }
                    let offset_mutate_chance_data = conn.offset_mutate_chance.to_be_bytes();
                    for k in 0..8 {
                        big_arr[x] = offset_mutate_chance_data[k];
                        x += 1;
                    }
                    let offset_mutate_amount_data = conn.offset_mutate_amount.to_be_bytes();
                    for k in 0..8 {
                        big_arr[x] = offset_mutate_amount_data[k];
                        x += 1;
                    }
                }
                let activation_thresh_data = conns.activation_thresh.to_be_bytes();
                for k in 0..8 {
                    big_arr[x] = activation_thresh_data[k];
                    x += 1;
                }
                let trait_swap_chance_data = conns.trait_swap_chance.to_be_bytes();
                for k in 0..8 {
                    big_arr[x] = trait_swap_chance_data[k];
                    x += 1;
                }
            }
        }

        file.write_all(&big_arr).expect("Failed to save model to file!");
    }
}
