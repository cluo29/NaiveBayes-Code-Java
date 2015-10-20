import weka.core.Attribute;
import java.util.ArrayList;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;


public class naive {
	
	public static void main(String[] args) throws Exception  {
	  /*InstanceQuery query = new InstanceQuery();
	  query.setUsername("nobody");
	  query.setPassword("");
	  query.setQuery("select * from whatsoever");
	  // You can declare that your data set is sparse
	  // query.setSparseData(true);
	  Instances data = query.retrieveInstances();
	  Instances test = query.retrieveInstances();
	  */
	  ArrayList<Attribute> atts = new ArrayList<Attribute>(2);
      ArrayList<String> classVal = new ArrayList<String>();
      classVal.add("A");
      classVal.add("B");
      atts.add(new Attribute("1"));
      atts.add(new Attribute("class",classVal));

      Instances dataRaw = new Instances("TestInstances",atts,0);
      
      double[] instanceValue1 = new double[dataRaw.numAttributes()];

      instanceValue1[0] = 1;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
 
      instanceValue1[0] = 1;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
  
      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
     
      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));

      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
   
      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));

      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
   
      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
     
      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));

      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
     
      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
      
      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));

      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
 
      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));

      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));

      instanceValue1[0] = 0;
      instanceValue1[1] = 0;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
      
      double[] instanceValue2 = new double[dataRaw.numAttributes()];


      instanceValue2[0] = 0;
      instanceValue2[1] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));

      instanceValue2[0] = 1;
      instanceValue2[1] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));
    
      instanceValue2[0] = 1;
      instanceValue2[1] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));
    
      instanceValue2[0] = 0;
      instanceValue2[1] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));
 
      instanceValue2[0] = 1;
      instanceValue2[1] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));
   
      instanceValue2[0] = 1;
      instanceValue2[1] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));

      instanceValue2[0] = 1;
      instanceValue2[1] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));

      instanceValue2[0] = 1;
      instanceValue2[1] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));

      instanceValue2[0] = 1;
      instanceValue2[1] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));
      dataRaw.setClassIndex(dataRaw.numAttributes()-1);
      
      
	  NaiveBayes model=new NaiveBayes();
	  
	  model.buildClassifier(dataRaw);   // build classifier
	  
	  Instances dataRaw2 = new Instances("EvalInstances",atts,0);
	  
	  double[] instanceValue3 = new double[dataRaw2.numAttributes()];

      instanceValue3[0] = 1;
      //instanceValue3[2] = 1;

      dataRaw2.add(new DenseInstance(1.0, instanceValue3));
      dataRaw2.setClassIndex(dataRaw2.numAttributes()-1);	  
	  

	  double[] x2 = model.distributionForInstance(dataRaw2.instance(0));
	  System.out.println(model.classifyInstance(dataRaw2.instance(0)));
	  
	  System.out.println(x2[0]);
	  System.out.println(x2[1]);
  }
}
