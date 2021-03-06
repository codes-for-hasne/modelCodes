import json
import numpy as np
from codes.utils.metric import Metric
from codes.utils.filewriter import write_to_file




def metric(params):
    """
    metric functions
    :param params:
    :return:
    """
    js = json.loads(open(params["res_input"]).read())
    # tar = js['target']
    js = js['embedding']

    embedding = np.array(js)

    res_path = params["metric_output"]

    ret = []
    for metric in params["metric_function"]:
        if metric['metric_func'] == "draw_circle_2D":
            # pic_path = os.path.join(PIC_PATH, "draw_circle_" + str(int(time.time() * 1000.0)) + ".pdf")
            # dh.symlink(pic_path, os.path.join(PIC_PATH, "new_draw_circle"))
            # getattr(Metric, metric["metric_func"])(coordinates, radius, metric, params["num_nodes"], pic_path)
            pass
        elif metric['metric_func'] == 'classification':
            res = getattr(Metric, metric["metric_func"])(embedding, metric)
            ret.append((metric["metric_func"], res))
        elif metric['metric_func'] == 'display':
            Metric.display(embedding)
        elif metric['metric_func'] == 'parseData':
            Metric.parseData()
        elif metric['metric_func'] == 'displayTrainRes':
            Metric.displayTrainRes(embedding)
        elif metric['metric_func'] == 'visualization':
            Metric.visualization(embedding)
        elif metric['metric_func'] == 'visualizationZoom':
            Metric.visualizationZoom(embedding)
        elif metric['metric_func'] == 'drawG':
            Metric.drawG()
        elif metric['metric_func'] == 'poincare':
            Metric.poincareDraw()
        elif metric['metric_func'] == 'drawAS':
            Metric.drawAS()
        elif metric['metric_func'] == 'drawPoincare':
            Metric.drawPoincare()
        elif metric['metric_func'] == 'drawGNE':
            Metric.drawGNE()
        elif metric['metric_func'] == 'reconstruction':
            Metric.reconstruction(embedding)
        elif metric['metric_func'] == 'drawHamiltonTree':
            Metric.drawHamiltonTree()





if __name__ == "__main__":

    inputFileName = 'path to the results'
    params = {
        'metric_function':[
            {
                'metric_func':'visualizationZoom'
            },
        ],
    }
    metric(params)

