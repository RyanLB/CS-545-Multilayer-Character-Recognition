//
//  Vector.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/25/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation
import Accelerate

enum VectorError : ErrorType {
    case MismatchedLength(expected: Int, found: Int)
}

class Vector {
    let length: Int
    private var _data: [Double]
    
    var data: [Double] {
        get {
            return _data
        }
    }
    
    init(length: Int) {
        self.length = length
        _data = [Double](count: length, repeatedValue: 0.0)
    }
    
    init(data: [Double]) {
        self._data = data
        length = data.count
    }
    
    subscript(element: Int) -> Double {
        get {
            assert(element >= 0 && element < length, "Index out of range")
            return _data[element]
        }
        set {
            assert(element >= 0 && element < length, "Index out of range")
            _data[element] = newValue
        }
    }
    
    func add(v2: Vector, scale: Double?) throws {
        guard v2.length == length else {
            throw VectorError.MismatchedLength(expected: length, found: v2.length)
        }
        cblas_daxpy(Int32(length), scale != nil ? scale! : 1.0, v2.data, 1, &_data, 1)
    }
    
    class func hadamardProduct(v1: Vector, v2: Vector) throws -> Vector {
        guard v2.length == v1.length else {
            throw VectorError.MismatchedLength(expected: v1.length, found: v2.length)
        }
        
        return Vector(data: zip(v1.data, v2.data).map{ $0.0 * $0.1 } )
    }
    
    class func add(v1: Vector, scale: Double?, v2: Vector) throws -> Vector{
        guard v1.length == v2.length else {
            throw VectorError.MismatchedLength(expected: v1.length, found: v2.length)
        }
        
        let newVector = Vector(data: v2.data)
        try newVector.add(v1, scale: scale)
        
        return newVector
    }
    
    class func randomVector(length: Int) -> Vector {
        assert(length >= 0, "Length cannot be negative")
        var randomData = [Double]()
        for _ in 0..<length {
            randomData.append(smallRandom())
        }
        
        return Vector(data: randomData)
    }
    
    ///Generates a small random number between -.25 and .25.
    
    private class func smallRandom() -> Double {
        let randVal = drand48() * 0.25
        if arc4random_uniform(2) == 0 {
            return -randVal
        }
        
        return randVal
    }

}