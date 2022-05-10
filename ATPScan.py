
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ScrumPy.Data import DataSets


class ScanRes(DataSets.DataSet):

    def __init__(self, *args, **kwargs):
        DataSets.DataSet.__init__(self, *args, **kwargs)


    def changers(self, lim=1e-7):
        """Filter list to select reactions which pass the cutoff for change in flux."""
        return [cname for cname in self.cnames if np.ptp(self.GetCol(cname)) > lim]


def check_ATP_modes(model, elmodes, ATPase="ATPase", summary=True):
    """
    Identify elementary modes producing ATP from nothing.

        Parameters:
            model (obj) : model
            elmodes (obj) : elementary modes
            ATPase (str) : ATPase reaction
            summary (bool) : print summary

        Returns:
            rv (list) : list of elementary modes producing ATP from nothing
    """

    transporters = list(filter(lambda x : x.endswith("_tx"), model.sm.cnames))
    modes = list(elmodes.ModesOf(ATPase).keys())
    rv = [mode for mode in modes if not SetUtils.intersect(transporters, list(elmodes.ReacsOf(mode).keys()))]

    if summary:
        log.info(f"ATP producing modes: {', '.join(rv)}")
    return rv


def scan(model: object, low=1.0, high=200.0, n=100, lp=None, flux_bounds=None, ATPase="ATPSynth", plot=True) -> (object, dict):
    """
    Perform an ATP scan on a Linear Programme.

        Parameters:
            model (obj) : model
            low (float) : lowest flux point
            high (float): highest flux point
            n (int) : number of points
            lp (obj) : linear programme object
            ATPase (str) : ATPase reaction
            flux_bounds (dict) : define flux bounds for lp
            plot (bool) : plot result using GNUplot

        Returns:
            rv (Object) : GNU plotting instance
            results (dict) : result of ATP scan [flux, reaction name, value]
    """
    if plot:
        rv = ScanRes()

    if not lp: 
        lp = model.GetLP()
        lp.SetObjective(model.sm.cnames)

    if flux_bounds:
        lp.SetFluxBounds(flux_bounds)

    step = (high - low) / (n-1)

    result = {}
    result["flux"] = result["reactions"] = result["values"] = None

    for atp in np.arange(low, high, step):

        lp.SetFixedFlux({ATPase: atp})
        lp.Solve(PrintStatus=False) 

        if lp.IsStatusOptimal():

            sol = lp.GetPrimSol()

            r, v = zip(*sol.items())
            f = [atp] * len(r)

            for key, sub, typ in zip(list(result.keys()), (f, r, v), (float, str, float)):
                master = result[key]
                if master is not None:
                   master = np.concatenate((master, np.array(sub, dtype=np.dtype(typ))), axis=None)
                else:
                    master = np.array(sub, dtype=np.dtype(typ))
                
                result[key] = master

            if plot: 
                rv.UpdateFromDic(sol)

    if plot:
        rv.SetPlotX(ATPase)
        rv.AddToPlot(rv.changers())
    
    return rv, result


def plot_atoms(result: dict, model, db):
    df = pd.DataFrame(result)
    df = df[df["reactions"].str.endswith("_tx")]

    transporter = df.reactions.drop_duplicates()\
        .values.tolist()

    carbon_atoms = {}
    nitrogen_atoms = {}

    for i in transporter:
        product = model.smx.Products(i)[0]
        carbons = db.dbs["Compound"][product].NumAtoms('C')
        nitrogens = db.dbs["Compound"][product].NumAtoms('N')
        carbon_atoms[i] = carbons
        nitrogen_atoms[i] = nitrogens

    df["Carbon"] = df.apply(lambda x : x["values"] * carbon_atoms[x["reactions"]], axis=1)
    df["Nitrogen"] = df.apply(lambda x : x["values"] * nitrogen_atoms[x["reactions"]], axis=1)

    f, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    sns.lineplot(data=df, x='flux', y='Carbon', hue='reactions', style='reactions', ax=ax1)
    sns.lineplot(data=df, x='flux', y='Nitrogen', hue='reactions', style='reactions', ax=ax2)
    ax2.legend([],[], frameon=False)
    plt.show()


def plot(result: dict):
    df = pd.DataFrame(result)
    sns.lineplot(data=result, x='flux', y='values', hue='reactions', style='reactions')
    plt.show()


def plot_transporters(rv, suffix="_tx"):
    transporter = [tx for tx in rv.cnames if tx.endswith(suffix)]
    rv.RemoveAllFromPlot()
    rv.AddToPlot(transporter)
    return rv


def abs_total_sum(sol):
    "Get total flux of LP solution"
    return sum([abs(a) for a in sol.values()])