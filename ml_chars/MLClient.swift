//
//  MLClient.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/27/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

enum MLClientError : ErrorType {
    case InvalidFilepath(path: String)
}

class MLClient {
    var trainingData = [Letter]()
    var testData = [Letter]()
    
    /// Wrapper function to load training data from a  file.
    func loadTrainingData(fromPath: String) throws {
        trainingData = try loadLetters(fromPath)
    }
    
    /// Wrapper function to load test data from a file.
    func loadTestData(fromPath: String) throws {
        testData = try loadLetters(fromPath)
    }
    
    func testWithParameters(hiddenNodes: Int, learningRate: Double, momentum: Double, epochLimit: Int) {
        let network = NeuralNetwork(hiddenLayerWidth: hiddenNodes)
        
        // I'm a bad programmer who ignores exceptions
        let history = try! network.train(trainingData, testData: testData, learningRate: learningRate, momentum: momentum, epochLimit: epochLimit)
        
        
    }
    
    /**
     Loads an array of Letter objects from a file at the given path.
     
     - Parameter fromPath: The relative path of the file from which to load Letters.
     
     - Returns: A list of Letter objects.
     */
    private func loadLetters(fromPath: String) throws -> [Letter] {
        let inputHandle = NSFileHandle(forReadingAtPath: fromPath)
        
        if inputHandle == nil {
            throw MLClientError.InvalidFilepath(path: fromPath)
        }
        
        var results = [Letter]()
        
        repeat {
            if let line = inputHandle?.getASCIILine() {
                let letter = Letter.init(withCommaSeparatedAttributeList: line)
                results.append(letter)
            }
            else {
                break
            }
            
        } while true
        
        inputHandle!.closeFile()
        
        return results
    }
}