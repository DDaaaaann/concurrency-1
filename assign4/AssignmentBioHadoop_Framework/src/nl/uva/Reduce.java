package nl.uva;

import java.io.IOException;
import java.util.Iterator;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;

public class Reduce extends MapReduceBase implements Reducer<ScoreWritable, Text, IntWritable, Text> {

    private int numberOfrelative = 1;
    private int counter = 0;
    Log log = LogFactory.getLog(Reduce.class);

    @Override
    public void configure(JobConf job) {
        numberOfrelative = job.getInt(AssignmentMapreduce.NUMBER_OF_RELATIVE_NAME, 5);
    }

    @Override
    public void reduce(ScoreWritable key, Iterator<Text> valueItrtr, OutputCollector<IntWritable, Text> output, Reporter rprtr) throws IOException {
        //Each Reducer receives the key value pairs from the Mappers, and 
        //emits them in the output. Here we don't do any commutation since the
        //keys, which are the scores come in already sorted. We just stop 
        //When we have the Nth highest 
        while (valueItrtr.hasNext() && counter < numberOfrelative) {
            counter++;
            
            //Here you only have to write the results to the output
            //Use the score as key and the sequence as value. 
            
			output.collect(key, valueItrtr.next());
			
            //Report progress
            rprtr.incrCounter(nl.uva.Map.Counters.INPUT_SEQUENCES, 1);
        }
        rprtr.setStatus("Found the " + numberOfrelative + " most similar sequences");
    }
}
