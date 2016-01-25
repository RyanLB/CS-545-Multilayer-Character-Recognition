//
//  SigmoidLayer.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/23/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation
import Accelerate

class SigmoidLayer {
    let nodeCount: Int
    let previousNodeCount: Int
    var biasVector: [Double]
    var weightMatrix: [[Double]]
    
    init(nodeCount: Int, previousNodeCount: Int) {
        self.nodeCount = nodeCount
        self.previousNodeCount = previousNodeCount
        biasVector = []
        weightMatrix = [[Double]](count: nodeCount, repeatedValue: [])
        
        for i in 0..<nodeCount {
            biasVector.append(SigmoidLayer.smallRandom())
            for j in 0..<previousNodeCount {
                weightMatrix[i][j] = SigmoidLayer.smallRandom()
            }
        }
    }
    
    
    func computeOutputs(fromInputs: [Double]) -> [Double] {
        var result = [Double](count: nodeCount, repeatedValue: 0.0)
        cblas_dcopy(Int32(nodeCount), biasVector, 1, &result, 1)
        
        // Well, my code is completely unreadable, but I blame Intel for making this CBLAS API
        cblas_dgemv(CblasRowMajor,            // Data is in row-major order
                    CblasNoTrans,             // Use the raw matrix instead of the transpose
                    Int32(nodeCount),         // Row count
                    Int32(previousNodeCount), // Column count
                    1.0,                      // Matrix scale factor
                    weightMatrix[0],          // Input matrix
                    Int32(nodeCount),         // Size of first dimension? Is this not just the same as the row count? wtf
                    fromInputs,               // Vector by which to multiply matrix
                    1,                        // Stride in X
                    1.0,                      // Scale for added vector
                    &result,                  // Vector to add
                    1                         // Stride in Y
                    )
            
        
        return result.map{ SigmoidLayer.sigmoid($0) }
    }
    
    ///Generates a small random number between -.25 and .25.
    
    private class func smallRandom() -> Double {
        let randVal = drand48() * 0.25
        if arc4random_uniform(2) == 0 {
            return -randVal
        }
        
        return randVal
    }
    
    private class func matrixCopy(source: [[Double]]) -> [[Double]] {
        let cols = source[0].count
        var output = [[Double]](count: source.count, repeatedValue: [Double](count: cols, repeatedValue: 0.0))
        
        // Use a CBLAS function to perform a vectorized copy of each row
        for i in 0..<source.count {
            var row = output[i]
            cblas_dcopy(Int32(cols), source[i], 1, &row, 1)
        }
        
        return output
    }
    
    private class func sigmoid(input: Double) -> Double {
        return 1.0 / (1.0 + pow(M_E, -input))
    }
}