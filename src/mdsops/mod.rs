
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


pub mod mds_ops{
    use rand::rngs::ThreadRng;
    use crate::{Algopar, Problem};
    use rand::Rng;
    use crate::mdsobj::mds_obj;

    pub fn init(probparams: &Problem, random: &mut ThreadRng) -> Vec<Vec<f32>> {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for _ in 0..(probparams.dim + 1) {
            let initsimplex: Vec<f32> = (probparams.xmin).iter()
                .zip((probparams.xmax).iter())
                .map(|(&a, &b)| {
                    let random_number: f32 = random.random_range(0.0..1.0);
                    a + random_number * (b - a)
                })
                .collect();
            result.push(initsimplex);
        }
        result
    }


    pub fn check_bounds(probparams: &Problem, simplex: &Vec<Vec<f32>>) -> Vec<Vec<f32>>  {
        let max = probparams.xmax[0];
        let min = probparams.xmin[0];
        let mut boundedsimplex = simplex.clone();
        for point in boundedsimplex.iter_mut() {
            for value in point.iter_mut() {
                if *value > max {
                    *value = max;
                }
                if *value < min {
                    *value = min;
                }
            }
        }
        boundedsimplex
    }
    
    pub fn sortsimplex(simplex: &Vec<Vec<f32>>, fsimplex: &Vec<f32>) -> (Vec<Vec<f32>>, Vec<f32>){
        // Create a vector of tuples
        let sortingsimplex = simplex.clone();
        let sortingfsimplex = fsimplex.clone();
        let mut preparevecs: Vec<(f32, Vec<f32>)> = sortingfsimplex.iter().cloned().zip(sortingsimplex).collect();

        // Sort the paired vector by the first element of the tuple
        preparevecs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Unzip the sorted pairs back into two vectors
        let (sorted_fsimplex, sorted_simplex): (Vec<_>, Vec<_>) = preparevecs.into_iter().unzip();
        (sorted_simplex, sorted_fsimplex)
    }


    pub fn reflection(probparams: &Problem, simplex: &Vec<Vec<f32>>, fevs: &mut usize) -> (Vec<Vec<f32>>, Vec<f32>){
        let mut refsimplex: Vec<Vec<f32>> = Vec::new();
        for i in 0..simplex.len() {
            let tmprefsimplex: Vec<f32> = simplex[0]
                .iter()
                .zip(&simplex[i])
                .map(|(a, b)| a - (b - a))
                .collect();
            refsimplex.push(tmprefsimplex);
        };
        refsimplex = check_bounds(&probparams, &mut refsimplex);
        let mut frsimplex = mds_obj::objeval(&probparams, &refsimplex);
        *fevs += probparams.dim+1;
        (refsimplex,frsimplex) = sortsimplex(&mut refsimplex, &mut frsimplex);
        (refsimplex,frsimplex)
    }
    
    pub fn expansion(probparams: &Problem, simplex: &Vec<Vec<f32>>, fevs: &mut usize, parameters: &Algopar) -> (Vec<Vec<f32>>, Vec<f32>){
        let mut expasimplex: Vec<Vec<f32>> = Vec::new();
        for i in 0..simplex.len() {
            let tmpexpasimplex: Vec<f32> = simplex[0]
                .iter()
                .zip(&simplex[i])
                .map(|(a, b)| a - parameters.pexp*(b - a))
                .collect();
            expasimplex.push(tmpexpasimplex);
        };
        expasimplex = check_bounds(&probparams, &mut expasimplex);
        let mut fexpasimplex = mds_obj::objeval(&probparams, &expasimplex);
        *fevs += probparams.dim+1;
        (expasimplex,fexpasimplex) = sortsimplex(&mut expasimplex, &mut fexpasimplex);
        (expasimplex,fexpasimplex)
    }
    
    pub fn contraction(probparams: &Problem, simplex: &Vec<Vec<f32>>, fevs: &mut usize, parameters: &Algopar)-> (Vec<Vec<f32>>, Vec<f32>) {
        let mut contrsimplex: Vec<Vec<f32>> = Vec::new();
        for i in 0..simplex.len() {
            let tmpcontrsimplex: Vec<f32> = simplex[0]
                .iter()
                .zip(&simplex[i])
                .map(|(a, b)| a - parameters.pcontr*(b - a))
                .collect();
            contrsimplex.push(tmpcontrsimplex);
        };
        contrsimplex = check_bounds(&probparams, &mut contrsimplex);
        let mut fcontrsimplex = mds_obj::objeval(&probparams, &contrsimplex);
        *fevs += probparams.dim+1;
        (contrsimplex,fcontrsimplex) = sortsimplex(&mut contrsimplex, &mut fcontrsimplex);
        (contrsimplex,fcontrsimplex)
    }
}