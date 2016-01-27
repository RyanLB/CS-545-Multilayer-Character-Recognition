//
//  HiddenLayer.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/24/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

class HiddenLayer : SigmoidLayer {
    func calculateErrors(outputs: Vector, nextLayerWeightedErrors: Vector) -> Vector {
        return Vector(data: zip(outputs.data, nextLayerWeightedErrors.data).map{ $0.0 * (1 - $0.0) * $0.1 })
    }
}