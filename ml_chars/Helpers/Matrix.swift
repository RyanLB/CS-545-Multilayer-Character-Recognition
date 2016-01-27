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
    
    init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        
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
    
    init(rowVectors: [Vector]) throws {
        rows = rowVectors.count
        cols = rowVectors[0].length
        
        _dataArray = rowVectors
        
        for v in rowVectors {
            guard v.length == cols else {
                throw MatrixError.NonRectangularData
            }
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
    
    /**
     Multiplies this Matrix by a given Vector.
     
     - Parameter v: The vector by which to multiply this Matrix. Not altered.
     
     - Returns: The resulting vector.
     */
    func vectorProduct(v: Vector) throws -> Vector {
        guard v.length == cols else {
            throw VectorError.MismatchedLength(expected: cols, found: v.length)
        }
        
        return Vector(data: _dataArray.map({ (row: Vector) -> Double in cblas_ddot(Int32(rows), row.data, 1, v.data, 1) }))
    }
    
    /**
     Performs element-wise addition of this Matrix to another Matrix.
     
     This matrix's values are replaced with the results.
     
     - Throws: `MatrixError.MismatchedHeight` or `MatrixError.MismatchedWidth` if `m` has different dimensions.
     */
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
    
    /// Returns a new Matrix representing the transpose of this one.
    func transpose() -> Matrix {
        var newData = [[Double]]()
        
        for i in 0..<cols {
            newData.append(_dataArray.map{ $0[i] })
        }
        
        return try! Matrix(data: newData)
    }
    
    func scale(scaleFactor: Double) {
        for v in _dataArray {
            v.scale(scaleFactor)
        }
    }
    
    class func scaled(m: Matrix, scaleFactor: Double) -> Matrix {
        let newMat = try! Matrix(data: m._dataArray.map{ $0.data })
        newMat.scale(scaleFactor)
        return newMat
    }
    
    /// Creates and returns a Matrix of small (-.25 < x < .25) random values with the given dimensions.
    class func randomMatrix(rows: Int, columns: Int) -> Matrix {
        assert(rows >= 0 && columns >= 0, "Dimensions cannot be negative")
        
        var randomData = [Vector]()
        for _ in 0..<rows {
            randomData.append(Vector.randomVector(columns))
        }
        
        return try! Matrix(rowVectors: randomData)
    }
}