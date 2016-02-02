//
//  AccuracyHistory.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/31/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

class AccuracyHistory {
    struct AccuracyPair {
        let trainingAccuracy: Double
        let testAccuracy: Double
        init(trainingAccuracy: Double, testAccuracy: Double) {
            self.trainingAccuracy = trainingAccuracy
            self.testAccuracy = testAccuracy
        }
    }
    
    var history = [AccuracyPair]()
    
    /// Appends an entry to this history.
    func add(p: AccuracyPair) {
        history.append(p)
    }
    
    /// Writes this history to a file at the given location.
    func writeToCSV(filename: String) {
        NSFileManager.defaultManager().createFileAtPath(filename, contents: nil, attributes: nil)
        let fh = NSFileHandle(forWritingAtPath: filename)
        
        // This would probably be better done with a guard statement, but I am too lazy to write an error type
        assert(fh != nil, "Invalid filepath: \(filename)")
        let handle = fh!
        
        for p in history {
            handle.writeData("\(p.trainingAccuracy),\(p.testAccuracy)\n".dataUsingEncoding(NSUTF8StringEncoding)!)
        }
        
        handle.closeFile()
    }
}