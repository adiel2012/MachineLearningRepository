package corenpl1;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import java.util.*;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        if(args == null || args.length == 0)
        {
            System.out.println("Empty parameters");
            return;
        }
        // build in this way   mvn clean compile assembly:single
        //   java -jar target\acmartifactid-1.0-SNAPSHOT-jar-with-dependencies.jar
        init();
        String[] sentences = args;

        for(String str : sentences)
        {
            System.out.println(str + "----->" + String.valueOf(findSentiment(str)) );
        }

        System.out.println( "--------------------" );
    }



    static StanfordCoreNLP pipeline;
 
    public static void init() {

         // set up pipeline properties
        Properties props = new Properties();
        // set the list of annotators to run
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,kbp,quote");
        // set a property for an annotator, in this case the coref annotator is being set to use the neural algorithm
        props.setProperty("coref.algorithm", "neural");
        // build pipeline
        pipeline = new StanfordCoreNLP(props);

        //pipeline = new StanfordCoreNLP("nlp.properties");
        System.out.println("End read pipeline");
    }
  
    public static int findSentiment(String text) {
        System.out.println(text);
        System.out.println(pipeline.getClass());
        int mainSentiment = 0;
        if (text != null && text.length() > 0) {
            int longest = 0;
            Annotation annotation = pipeline.process(text);
            List<CoreMap> ops = annotation.get(CoreAnnotations.SentencesAnnotation.class);
            if(ops != null)
            for (CoreMap sentence : ops) {
                if(sentence == null)
                    continue;
                Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
                if(tree == null)
                    continue;
                int sentiment = RNNCoreAnnotations.getPredictedClass(tree);                
                String partText = sentence.toString();
                
                if (partText.length() > longest) {
                    
                    mainSentiment = sentiment;
                    longest = partText.length();
                }
 
            }
        }
        return mainSentiment;
    }
}
