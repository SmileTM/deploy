import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.Arrays;

public class LoadBertModelByDJL {
    public static void main(String[] args) throws TranslateException, MalformedModelException, ModelNotFoundException, IOException {
        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelUrls("src/main/resources/testBert")
                .optEngine("TensorFlow")
                .optDevice(Device.cpu())
                .optProgress(new ProgressBar())
                .build();
        ZooModel<NDList, NDList> model1 = criteria.loadModel();
        Predictor<NDList, NDList> predictor = model1.newPredictor();
        System.out.println(predictor);

        NDManager manager = NDManager.newBaseManager();
        long[] raw_data = new long[512];
        for (int i = 0; i < 512; i++) {
            raw_data[i] = 1;
        }
        NDArray data = manager.create(raw_data).reshape(1, 512);
        NDList inputs = new NDList(data, data, data);
        NDList res = predictor.predict(inputs);
        System.out.println();

        System.out.println("===========");
        System.out.println("output_1 输出形状为：" + res.get("output_1").getShape());
        System.out.println("===========");
        System.out.println("预测结果为：" + Arrays.toString(res.get("output_1").toArray()));

        Long start = System.currentTimeMillis();
        for (int i = 0; i < 50; i++) {
            predictor.predict(inputs);
        }
        Long end = System.currentTimeMillis();
        System.out.println("循环50次 耗时：");
        System.out.println((end - start) / 1000.);


    }
}
