from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heu_miner
from pm4py.algo.discovery.inductive import algorithm as ind_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_factory
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.visualization.dfg import visualizer as dfg_vis
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

XES_PATH = "/home/zzh/data_lab/hw3/output_simplified.xes"
OUT_DIR = "/home/zzh/data_lab/hw3"


def save_petri(net, im, fm, png_path, pnml_path=None):
    gviz = pn_vis.apply(net, im, fm)
    pn_vis.save(gviz, png_path)
    if pnml_path:
        pnml_exporter.apply(net, im, pnml_path, final_marking=fm)


def main():
    log = xes_importer.apply(XES_PATH)

    net_a, im_a, fm_a = alpha_miner.apply(log)
    save_petri(net_a, im_a, fm_a, f"{OUT_DIR}/alpha_petri.png", f"{OUT_DIR}/alpha_petri.pnml")

    net_h, im_h, fm_h = heu_miner.apply(log)
    save_petri(net_h, im_h, fm_h, f"{OUT_DIR}/heuristics_petri.png")

    tree_i = ind_miner.apply(log)
    net_i, im_i, fm_i = pt_converter.apply(tree_i)
    save_petri(net_i, im_i, fm_i, f"{OUT_DIR}/inductive_petri.png")

    dfg = dfg_factory.apply(log)
    gviz_dfg = dfg_vis.apply(dfg, log=log)
    dfg_vis.save(gviz_dfg, f"{OUT_DIR}/dfg.png")


if __name__ == "__main__":
    main()

