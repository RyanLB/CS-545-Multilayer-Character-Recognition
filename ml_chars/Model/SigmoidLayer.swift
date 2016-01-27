//
//  SigmoidLayer.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/23/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

enum SigmoidLayerError : ErrorType {
    case InputVectorLengthMismatch(expected: Int, found: Int)
}

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
    
    /**
     Calculates the output from this layer, `\sigma (WV + B)`
     
     - Parameter inputs: The outputs from the preceding layer.
     
     - Returns: The resulting vector.
     */
    func calculateOutput(inputs: Vector) throws -> Vector {
        assert(inputs.length == inputCount, "Input vector has incorrect length")
        
        let result = try weights.vectorProduct(inputs)
        try result.add(biases, scale: nil)
        
        result.sigmoidTransform()
        return result
    }
    
    func weightedErrors(errors: Vector) throws -> (weightErrors: Vector, biasErrors: Vector) {
        let weightErrors = try weights.transpose().vectorProduct(errors)
        let biasErrors = try Vector.hadamardProduct(biases, v2: errors)
        
        return (weightErrors: weightErrors, biasErrors: biasErrors)
    }
    
    private class func sigmoid(input: Double) -> Double {
        return 1.0 / (1.0 + pow(M_E, -input))
    }
}