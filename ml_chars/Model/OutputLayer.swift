//
//  OutputLayer.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/24/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

class OutputLayer : SigmoidLayer {
    override init(weightCount: Int, inputCount: Int) {
        super.init(weightCount: weightCount, inputCount: inputCount)
    }
    
    init(original: OutputLayer) {
        super.init(weights: original.weights.data.map{ $0.data }, biases: original.biases.data)
    }
    
    func calculateErrors(outputs: Vector, targets: Vector) -> Vector {
        return Vector(data: zip(outputs.data, targets.data).map{ $0.0 * (1 - $0.0) * ($0.1 - $0.0) })
    }
}