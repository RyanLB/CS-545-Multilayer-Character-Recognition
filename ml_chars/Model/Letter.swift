//
//  Letter.swift
//  Perceptron Character Recognition
//
//  Created by Ryan Bernstein on 1/7/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

class Letter {
    var knownValue: Character
    var attributeVector: Array<Double>
    
    init(fromKnownValue: Character, withAttributes: Array<Double>) {
        knownValue = fromKnownValue
        attributeVector = withAttributes
    }
    
    convenience init(withCommaSeparatedAttributeList: String) {
        let splitAttributes = withCommaSeparatedAttributeList.characters.split{$0 == ","}.map{String($0)}
        
        let knownChar = splitAttributes[0][splitAttributes[0].startIndex]
        
        let tail = Array(splitAttributes[1..<splitAttributes.count])
        let attributeArray:[Double] = tail.map{ Double($0)! / 15 }
        
        self.init(fromKnownValue: knownChar, withAttributes: attributeArray)
    }
}