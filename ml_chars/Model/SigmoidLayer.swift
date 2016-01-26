//
//  SigmoidLayer.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/23/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

class SigmoidLayer {
    let weightCount: Int
    let inputCount: Int
    
    var weights: Matrix
    var biases: Vector
    
    init(weightCount: Int, inputCount: Int) {
        self.weightCount = weightCount
        self.inputCount = inputCount
        
        biases = Vector.randomVector(weightCount)
        weights = Matrix.randomMatrix(weightCount, columns: inputCount)
    }
    
    private class func sigmoid(input: Double) -> Double {
        return 1.0 / (1.0 + pow(M_E, -input))
    }
}