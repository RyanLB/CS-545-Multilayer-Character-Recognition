//
//  NSArrayExtension.swift
//  perceptron_chars
//
//  Created by Ryan Bernstein on 1/11/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

extension Array {
    /**
     Performs a Fisher-Yates shuffle on this array.
     */
    mutating func shuffle() {
        for i in 0..<(self.count - 1) {
            let j = Int(arc4random_uniform(UInt32(self.count - i)))
            let tmp = self[i]
            self[i] = self[i + j]
            self[i + j] = tmp
        }
    }
    
    /**
     Returns the indices of all elements that satisfy the given predicate.
     */
    func indicesWhere(predicate: Element -> Bool) -> [Int] {
        var results = [Int]()
        
        for i in 0..<self.count {
            if predicate(self[i]) {
                results.append(i)
            }
        }
        
        return results
    }
}