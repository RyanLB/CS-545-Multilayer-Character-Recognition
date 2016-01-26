//
//  Matrix.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/25/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation
import Accelerate

enum MatrixError : ErrorType {
    case NonRectangularData
    case MismatchedHeight(expected: Int, found: Int)
    case MismatchedWidth(expected: Int, found: Int)
}

class Matrix {
    let rows: Int
    let cols: Int
    private var _dataArray: [Vector]
    
    var data: [Vector] {
        get {
            return _dataArray
        }
        
        set {
            
        }
    }
    
    init(rows: UInt, cols: UInt) {
        self.rows = Int(rows)
        self.cols = Int(cols)
        
        self._dataArray = [Vector]()
        for _ in 0..<rows {
            _dataArray.append(Vector(length: self.cols))
        }
    }
    
    init(data: [[Double]]) throws {
        rows = data.count
        cols = data[0].count
        
        _dataArray = [Vector]()
        for i in 0..<rows {
            guard data[i].count == cols else {
                throw MatrixError.NonRectangularData
            }
            _dataArray.append(Vector(data: data[i]))
        }
    }
    
    subscript(row: Int) -> Vector {
        get {
            return _dataArray[row]
        }
    }
    
    subscript(row: Int, column: Int) -> Double {
        get {
            return _dataArray[row][column]
        }
    }
    
    func vectorMultiply(v: Vector) throws -> Vector {
        guard v.length == cols else {
            throw VectorError.MismatchedLength(expected: cols, found: v.length)
        }
        
        return Vector(data: _dataArray.map({ (row: Vector) -> Double in cblas_ddot(Int32(rows), row.data, 1, v.data, 1) }))
    }
    
    func matrixAdd(m: Matrix) throws {
        guard rows == m.rows else {
            throw MatrixError.MismatchedHeight(expected: rows, found: m.rows)
        }
        
        guard cols == m.cols else {
            throw MatrixError.MismatchedWidth(expected: cols, found: m.cols)
        }
        
        for vecs in zip(_dataArray, m.data) {
            try vecs.0.add(vecs.1, scale: nil)
        }
    }
}