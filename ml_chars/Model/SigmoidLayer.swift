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

struct WeightedErrors {
    let weightErrors: Vector
    let biasErrors: Vector
    
    init(weightErrors: Vector, biasErrors: Vector) {
        self.weightErrors = weightErrors
        self.biasErrors = biasErrors
    }
}

struct Deltas {
    let weightDeltas: Matrix
    let biasDeltas: Vector
    
    init(weightDeltas: Matrix, biasDeltas: Vector) {
        self.weightDeltas = weightDeltas
        self.biasDeltas = biasDeltas
    }
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
    
    init(weights: [[Double]], biases: [Double]) {
        self.weights = try! Matrix(data: weights)
        self.biases = Vector(data: biases)
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
    
    func weightedErrors(errors: Vector) throws -> WeightedErrors {
        let weightErrors = try weights.transpose().vectorProduct(errors)
        let biasErrors = try Vector.hadamardProduct(biases, v2: errors)
        
        return WeightedErrors(weightErrors: weightErrors, biasErrors: biasErrors)
    }
    
    func train(learningRate: Double, inputs: Vector, errors: Vector, momentum: Double, previousDeltas: Deltas?) throws -> Deltas {
        let weightDeltas = try Matrix(rowVectors: errors.data.map{ Vector.scaled(inputs, scaleFactor: learningRate * $0) })
        
        let biasDeltas = Vector.scaled(errors, scaleFactor: learningRate)
        
        if let pDelts = previousDeltas {
            try weightDeltas.matrixAdd(Matrix.scaled(pDelts.weightDeltas, scaleFactor: momentum))
            try biasDeltas.add(pDelts.biasDeltas, scale: momentum)
        }
        
        try weights.matrixAdd(weightDeltas)
        try biases.add(biasDeltas, scale: nil)
        
        return Deltas(weightDeltas: weightDeltas, biasDeltas: biasDeltas)
    }
    
    private class func sigmoid(input: Double) -> Double {
        return 1.0 / (1.0 + pow(M_E, -input))
    }
}