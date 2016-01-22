import weka.core.Attribute;
import java.util.ArrayList;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.classifiers.trees.J48;
//Naive bayes

public class naive {
	public static void main(String[] args) throws Exception  {

		ArrayList<Attribute> attributes = new ArrayList<Attribute>(3);
		ArrayList<String> classVal = new ArrayList<String>();
		classVal.add("ok");
		classVal.add("cancel");
		classVal.add("cancel2");
		attributes.add(new Attribute("hour"));
		attributes.add(new Attribute("minute"));
		attributes.add(new Attribute("class",classVal));

		Instances dataRaw = new Instances("TestInstances",attributes,0);
      
      double[] instanceValue1 = new double[3];

      instanceValue1[0] = 1;
      instanceValue1[1] = 0;
      instanceValue1[2] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
 
      instanceValue1[0] = 1;
      instanceValue1[1] = 0;
      instanceValue1[2] = 1;
      dataRaw.add(new DenseInstance(1.0, instanceValue1));
  
      double[] instanceValue2 = new double[dataRaw.numAttributes()];

      instanceValue2[0] = 0;
      instanceValue2[1] = 1;
      instanceValue2[2] = 2;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));

      instanceValue2[0] = 1;
      instanceValue2[1] = 1;
      instanceValue2[2] = 2;
      dataRaw.add(new DenseInstance(1.0, instanceValue2));
      
      dataRaw.setClassIndex(dataRaw.numAttributes()-1);
      
	  NaiveBayes model=new NaiveBayes();
	  
	  model.buildClassifier(dataRaw);   // build classifier
	  
	  //model.updateClassifier(dataRaw.instance(22));
	  
	  Instances dataRaw2 = new Instances("EvalInstances",attributes,0);
	  
	  double[] instanceValue3 = new double[dataRaw2.numAttributes()];

      instanceValue3[0] = 1;
      instanceValue3[1] = 0;

      dataRaw2.add(new DenseInstance(1.0, instanceValue3));
      dataRaw2.setClassIndex(dataRaw2.numAttributes()-1);	  
	  

	  double[] x2 = model.distributionForInstance(dataRaw2.instance(0));
	  //model.updateClassifier(arg0);
	  
	  System.out.println(model.classifyInstance(dataRaw2.instance(0)));
	  
	  System.out.println(x2[0]);
	  System.out.println(x2[1]);
	  System.out.println(x2[2]);
	  
	  String[] options = new String[1];
	  options[0] = "-U";
	  J48 tree = new J48();
	  tree.setOptions(options); 
		tree.buildClassifier(dataRaw); 
		System.out.println(tree.classifyInstance(dataRaw2.instance(0)));
	  
	  
	}
}
