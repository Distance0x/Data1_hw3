from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.obj import Marking
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_eval
from pm4py.algo.evaluation.precision import algorithm as precision_eval
from pm4py.algo.evaluation.generalization import algorithm as gen_eval
from pm4py.algo.evaluation.simplicity import algorithm as simp_eval

XES_PATH = "/home/zzh/data_lab/hw3/output_simplified.xes"
PNML_MODELS = {
    "Alpha++_ProM": "/home/zzh/data_lab/hw3/Prom_data/alpha++.apnml",
    "Heuristics_ProM": "/home/zzh/data_lab/hw3/Prom_data/Heur.pnml",
    "Inductive_ProM": "/home/zzh/data_lab/hw3/Prom_data/Inductive.apnml",
}

# 针对缺失终止标记的模型，可在此指定 final marking（place 名 -> tokens）
CUSTOM_FINAL_MARKINGS = {
    "Heuristics_ProM": {"n2": 1, "n4": 1, "n20": 1},  # Consultation / Deliver Medicine / Report Completed 后置库所
}
FITNESS_VARIANT = fitness_eval.Variants.TOKEN_BASED
PRECISION_VARIANT = precision_eval.Variants.ETCONFORMANCE_TOKEN


def evaluate(name, net, im, fm, log):
    fit = fitness_eval.apply(log, net, im, fm, variant=FITNESS_VARIANT)
    prec = precision_eval.apply(log, net, im, fm, variant=PRECISION_VARIANT)
    gen = gen_eval.apply(log, net, im, fm)
    simp = simp_eval.apply(net)
    total_events = sum(len(t) for t in log)
    print(f"\n[{name}] traces={len(log)}, events={total_events}")
    print(f"  fitness({FITNESS_VARIANT.name}) -> {fit}")
    print(f"  precision({PRECISION_VARIANT.name}) -> {prec:.4f}")
    print(f"  generalization -> {gen:.4f}")
    print(f"  simplicity -> {simp:.4f}")


def main():
    log = xes_importer.apply(XES_PATH)
    print(f"日志: {XES_PATH}")

    for name, pnml_path in PNML_MODELS.items():
        print(f"\n载入 PNML: {pnml_path}")
        net, im, fm = pnml_importer.apply(pnml_path)
        if name in CUSTOM_FINAL_MARKINGS:
            fm = Marking({
                pl: tokens
                for pl in net.places
                for pid, tokens in CUSTOM_FINAL_MARKINGS[name].items()
                if pl.name == pid
            })
        evaluate(name, net, im, fm, log)


if __name__ == "__main__":
    main()

