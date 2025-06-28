
// Copyright 2025 Grigorios Piperagkas. All rights reserved.
// Use of this source code is governed by a BSD-3-clause
// license that can be found in the LICENSE file.
/*
//////////////////////////////////////////////////////////////////////////////

Main algorithm for Parallel Multidirectional Search proposed by VJ Torczon(1989).
implemented for evaluation of objective function in parallel on cuda enabled GPUs.

June 2025. 
/////////////////////////////////////////////////////////////////////////////

 */
mod mdsops;
mod mdsobj;

// use cuda::runtime::*;
// use rand::Rng;
use mdsops::mds_ops;
use mdsobj::mds_obj;

pub struct Algopar {
   pexp: f32,
   pcontr: f32,
}

// Problem parameters
pub struct Problem {
   dim: usize,
   bench: usize,
   //nfpar: u32,
   //maxfevs: u32,
   maxiters: u32,
   // e: u32,
   xmin: Vec<f32>,
   xmax: Vec<f32>,
   // Fparams: Vec<f32>,
   // Fiparams: Vec<u32>,
}

fn main() {


   //Bounds for benchmarking functions
   let xmini:Vec<f32> = vec![-100.0,-30.0,-5.12,-600.0,-20.0,-8.0,-10.0,-10.0,-4.0,-10.0,-10.0,-100.0,-65.536,-500.0];
   let xmaxi:Vec<f32> = vec![100.0,30.0,5.12,600.0,30.0,8.0,10.0,10.0,5.0,10.0,10.0,100.0,65.536,500.0];



       //Define problem and algorithms structures
   let parameters = Algopar {
       pexp: 2.0,
       pcontr: 0.5,
   };

   let mut probparams = Problem{
       maxiters: 10000000,
       //maxfevs: 10000000,
       //nfpar: 0,
       dim: 100,
       bench: 2,
       xmin: Vec::new(),
       xmax: Vec::new(),
   };

   if let Some(&boundmin) = xmini.get(probparams.bench) {
       probparams.xmin = vec![boundmin; probparams.dim];
   }
   if let Some(&boundmax) = xmaxi.get(probparams.bench) {
       probparams.xmax = vec![boundmax; probparams.dim];
   }

   //-------------------------
   // DEFINE ALL PROBLEM DATA
   //-------------------------

   let mut simplex: Vec<Vec<f32>>;
   let mut fsimplex: Vec<f32>;

   let mut refl: Vec<Vec<f32>>;
   let mut frefl: Vec<f32>;

   let mut expa: Vec<Vec<f32>>;
   let mut fexpa: Vec<f32>;

   let mut contr: Vec<Vec<f32>>;
   let mut fcontr: Vec<f32>;

   let mut fevs: usize = 0;
   let mut random = rand::rng();

   simplex = mds_ops::init(&probparams, &mut random);
   simplex = mds_ops::check_bounds(&probparams, &mut simplex);
   fsimplex = mds_obj::objeval(&probparams, &mut simplex);
   fevs += probparams.dim+1;
   (simplex,fsimplex) = mds_ops::sortsimplex(&mut simplex, &mut fsimplex);
   
   let mut iterations=0;    
   while iterations <  probparams.maxiters {
       //Reflection
       (refl, frefl) = mds_ops::reflection(&probparams, &simplex, &mut fevs);
       if frefl[0] < fsimplex[0]{
           (expa, fexpa) = mds_ops::expansion(&probparams, &simplex, &mut fevs, &parameters);
           if fexpa[0] < frefl[0]{
               simplex = expa.clone();
               fsimplex = fexpa.clone();
           }else{
               simplex = refl.clone();
               fsimplex = frefl.clone();
           } 
       }else{
           (contr, fcontr) = mds_ops::contraction(&probparams, &simplex, &mut fevs, &parameters);
           simplex = contr.clone();
           fsimplex = fcontr.clone();
       }
       iterations += 1;
       println!("Iterations count: {:?}", iterations);
       println!("Sorted simplex min objective function value: {:?}", fsimplex[0]);
       println!("Sorted simplex max objective function value: {:?}", fsimplex[probparams.dim]);
       println!("Objective function evaluations: {:?}", fevs);
       
   }
   println!("Final min objective function value: {:?}", fsimplex[0]);
   //println!("Final simplex: {:?}", simplex);
}



