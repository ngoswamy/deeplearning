/****************************************
 *  Author: Neeraj Goswamy
 *
 ****************************************/

// This code is inspired from examples that come with DL4J

package org.deeplearning4j;

//Standards Imports
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;


import java.io.File;

/**
 * A General Data Classification Example
 *
 * For columnar data
 *
 * @author Neeraj Goswamy
 *
 */
public class MultiLayerPerceptronInteractiveModule_Ask {


    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 100;   // No of rows should be more that batchsize
        int nEpochs = 50;
        //Readin Trained Model
    	MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("model_output.net");
        
       Scanner in = new Scanner(System.in);
       while(true) {
        
        double[][] response = new double[1][13];
        System.out.println("Evaluate model....");
        
        // Get Age
        System.out.println("Enter Age :");
        response[0][0] = Integer.parseInt(in.nextLine());
        if(response[0][0] == 0)
        	System.exit(0);
        
        // Get Work Class
        System.out.println("Enter Work Class 0-Private, 1- Self Employed, 3- Govt 4-Never Worked :");
        response[0][1] = Integer.parseInt(in.nextLine());
        
        // Get Final Weight (guess a random number)
        //System.out.println("Enter :");
        response[0][2] = ThreadLocalRandom.current().nextInt(50000,400000+1);
        
        // Get Education
        System.out.println("Enter 0 - Bachelors, 4 - Prof_school/School, 10-Masters, 13-Doctorate :");
        response[0][3] = Integer.parseInt(in.nextLine());
        
        // Get
        //System.out.println("Enter :");
        
       // System.out.println("Enter Education Level: ");
       // System.out.println("School-0, Bachelors-1, Masters-2, Docotrate-3 ");
        int resp = (int) response[0][3]; //Integer.parseInt(in.nextLine());
        if(resp == 0)
        	response[0][4] = 13;
        else if((resp == 10) || (resp == 13))
        	response[0][4] = 14;
        else   //school (guess a random number between 5 and 12)
        	response[0][4] = ThreadLocalRandom.current().nextInt(5,12+1);
        
        //Get Profession
        System.out.println("Enter Profession: ");
        System.out.println("Tech-support-0, Handicraft-1, Other(Services)-2, Sales-3, Manager/Executive-4, Professional-5" );
        System.out.println("Handlers-cleaners-6, Machine/Factory-7, Administrative-8, Unknown-14 ");
        System.out.println("Farming-fishing-9, Transport-10, House-Service-11, Police-12, Armed-Forces-13 ");
        response[0][5] = Integer.parseInt(in.nextLine());
        
        // Get Marital Status
        System.out.println("Enter Marital Status :");
        System.out.println("0-Married 1-Divorced, 2-Never-married, 3-Separated, 4-Widowed");
        response[0][6] = Integer.parseInt(in.nextLine());
        
        // Get Relationship
        System.out.println("Enter Relationship 0-wife, 2 Husband, 4-other 5-Unmarried  :");
        response[0][7] = Integer.parseInt(in.nextLine());
        
        // Get Race
        System.out.println("Enter Race 0-White, 1- Asian, 3- Other, 4-Black :");
        response[0][8] = Integer.parseInt(in.nextLine());
        
        // Get 
        System.out.println("Enter Female-0, Male - 1 :");
        response[0][9] = Integer.parseInt(in.nextLine());
        
        // Get Capital Gain (guess a random number)
        //System.out.println("Enter :");
        int capgain= ThreadLocalRandom.current().nextInt(1000,15000+1);
        if(capgain%2 == 1) 
        	capgain=0;
        response[0][10] = capgain;
        
        // Get Capital Loss (guess a random number)
        //System.out.println("Enter :");
        int caploss = ThreadLocalRandom.current().nextInt(1000, 2000+1);
        if(capgain == 0) {
        	if(caploss%2 == 1)
        		caploss=0;
        } else
        	caploss =0;
        	
        response[0][11] = caploss;
        
        // Get Hours per week
        System.out.println("Enter Hours-per-Week :");
        response[0][12] = Integer.parseInt(in.nextLine());
        
        INDArray arr = Nd4j.create(response);
        INDArray predicted = model.output(arr,false).get(NDArrayIndex.point(0));   ///.getRow(0).toString().trim();
        Double zeropercentage = predicted.getDouble(0);
        int zeroval = (int) (zeropercentage * 100);
        
        System.out.println("Salary is less than 50K "+ zeroval + "%");
        
       System.out.println("Salary is more than or equal to 50K "+(100-zeroval)+"%"); 
        
       }
        
    }
}
