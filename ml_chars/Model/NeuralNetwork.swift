//
//  NeuralNetwork.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/26/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

class NeuralNetwork {
    private var _outputLayer: OutputLayer
    private var _hiddenLayer: HiddenLayer // Just one for now. May extend with a list later
    
    var outputLayer: OutputLayer {
        get { return _outputLayer }
    }
    
    var hiddenLayer: HiddenLayer {
        get { return _hiddenLayer }
    }
    
    init(hiddenLayerWidth: Int) {
        _outputLayer = OutputLayer(weightCount: 26, inputCount: hiddenLayerWidth)
        _hiddenLayer = HiddenLayer(weightCount: hiddenLayerWidth, inputCount: 16)
    }
    
    /// Copy constructor, wheee
    init(original: NeuralNetwork) {
        _outputLayer = OutputLayer(original: original.outputLayer)
        _hiddenLayer = HiddenLayer(original: original.hiddenLayer)
    }
    
    /**
     Guesses which character a given Letter represents.
     
     - Throws: `VectorError.MismatchedLength` if inputs to some layer don't match the expected length.
     
     - Returns: The character with the highest output activation. Ties will be broken via random number generation,
       but are unlikely.
     */
    func test(l: Letter) throws -> Character {
        let result = try outputLayer.calculateOutput(hiddenLayer.calculateOutput(Vector(data: l.attributeVector))).data
        return guessFromActivations(result)
    }
    
    /**
     Guesses which character a given Letter represents, providing activation information for training.
     
     - Throws: `VectorError.MismatchedLength` if inputs to some layer don't match the expected length.
     
     - Returns: A tuple containing activations from the hidden layer, activations from the output layer, and the
       character with the highest activation. Ties will be broken via random number generation, but are unlikely.
     */
    private func testWithLayerActivations(l: Letter) throws -> (hiddenActivations: Vector, outputActivations: Vector, guess: Character) {
        let hiddenActivations = try hiddenLayer.calculateOutput(Vector(data: l.attributeVector))
        let outputActivations = try outputLayer.calculateOutput(hiddenActivations)
        let guess = guessFromActivations(outputActivations.data)
        
        return (hiddenActivations: hiddenActivations, outputActivations: outputActivations, guess: guess)
    }
    
    func train(trainingData: [Letter], testData: [Letter], learningRate: Double, momentum: Double, epochLimit: Int) throws -> [(trainingAccuracy: Double, testAccuracy: Double)] {
        var history = [(trainingAccuracy: Double, testAccuracy: Double)]()
        var trainingAccuracy = try accuracy(trainingData)
        history.append((trainingAccuracy: trainingAccuracy, testAccuracy: try accuracy(testData)))
        
        for _ in 0..<epochLimit {
            let newNetwork = try epochResult(trainingData, learningRate: learningRate, momentum: momentum)
            
            let newTrainingAccuracy = try newNetwork.accuracy(trainingData)
            history.append((trainingAccuracy: newTrainingAccuracy, testAccuracy: try newNetwork.accuracy(testData)))
            
            if newTrainingAccuracy <= trainingAccuracy {
                break
            }
            
            trainingAccuracy = newTrainingAccuracy
            
            self._hiddenLayer = newNetwork.hiddenLayer
            self._outputLayer = newNetwork.outputLayer

        }
        
        return history
    }
    
    private func accuracy(dataset: [Letter]) throws -> Double {
        return try Double(dataset.countWhere{ try self.test($0) == $0.knownValue }) / Double(dataset.count)
    }
    
    private func epochResult(trainingData: [Letter], learningRate: Double, momentum: Double) throws -> NeuralNetwork {
        let newNetwork = NeuralNetwork(original: self)
        
        var outputDeltas: Deltas? = nil
        var hiddenDeltas: Deltas? = nil
        
        try trainingData.forEach {
            let results = try newNetwork.testWithLayerActivations($0)
            
            if results.guess != $0.knownValue {
                // Factor this into a method?
                
                var targets = [Double](count: 26, repeatedValue: 0.1)
                targets[NeuralNetwork.AZIndexFromCharacter($0.knownValue)] = 0.9
                
                let outputErrors = newNetwork.outputLayer.calculateErrors(results.outputActivations, targets: Vector(data: targets))
                let weighted = try newNetwork.outputLayer.weightedErrors(outputErrors)
                let hiddenErrors = newNetwork.hiddenLayer.calculateErrors(results.hiddenActivations, nextLayerWeightedErrors: weighted.weightErrors)
                
                outputDeltas = try newNetwork.outputLayer.train(learningRate,
                                                                inputs: results.hiddenActivations,
                                                                errors: outputErrors,
                                                                momentum: momentum,
                                                                previousDeltas: outputDeltas)
                
                hiddenDeltas = try newNetwork.hiddenLayer.train(learningRate,
                                                                inputs: Vector(data: $0.attributeVector),
                                                                errors: hiddenErrors,
                                                                momentum: momentum,
                                                                previousDeltas: hiddenDeltas)
                
            }
        }
        
        return newNetwork
    }
    
    private func guessFromActivations(activations: [Double]) -> Character {
        let winners = activations.indicesWhere{ $0 == activations.maxElement() }
        let winnerIndex = Int(arc4random_uniform(UInt32(winners.count)))
        return NeuralNetwork.characterFromAZIndex(winners[winnerIndex])
    }
    
    private class func characterFromAZIndex(index: Int) -> Character {
        assert(index >= 0 && index <= 26, "Index must be between 0 and 26")
        return Character(UnicodeScalar(index))
    }
    
    private class func AZIndexFromCharacter(c: Character) -> Int {
        let s = String(c).unicodeScalars
        let val = Int(s[s.startIndex].value)
        return val - 65
    }
}