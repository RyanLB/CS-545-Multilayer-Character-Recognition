//
//  ConfusionMatrix.swift
//  perceptron_chars
//
//  Created by Ryan Bernstein on 1/11/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

class ConfusionMatrix {
    let data: [(input: Letter, guess: Character)]
    
    var accuracy: Double {
        get {
            let correctCount = data.filter({ $0.input.knownValue == $0.guess }).count
            return Double(correctCount) / Double(data.count)
        }
    }
    
    var matrix: [[Int]] {
        get {
            var results = [[Int]](count: 26, repeatedValue: [Int](count: 26, repeatedValue: 0))
            for example in data {
                ++results[charToASCII(example.input.knownValue) - 65][charToASCII(example.guess) - 65]
            }
            
            return results
        }
    }
    
    init(withData: [(input: Letter, guess: Character)]) {
        data = withData
    }
    
    /**
     Generates a representation of this matrix as a string of comma/newline delimited values.
     */
     
    func toCSVString() -> String {
        let mat = matrix
        var str = ""
        for i in 0..<mat.count {
            for j in 0..<(mat.count - 1) {
                str += "\(mat[i][j]),"
            }
            str += "\(mat[i][mat.count - 1])\n"
        }
        
        return str
    }
    
    /**
     Converts a character to its ASCII ordinal.
     */
    private func charToASCII(c: Character) -> Int {
        let s = String(c).unicodeScalars
        return Int(s[s.startIndex].value)
    }
}