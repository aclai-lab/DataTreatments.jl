using DataTreatments

using SoleData: Artifacts

# fill your Artifacts.toml file;
# Artifacts.fillartifacts()

natopsloader = Artifacts.NatopsLoader()
Xts, yts = Artifacts.load(natopsloader)

dt = DataTreatment(Xts, yts)

test1 = get_dataset(dt)
test2 = get_dataset(dt, dataframe=true)
test3 = get_dataset(
    dt,
    TreatmentGroup(aggrfunc=aggregate(
        features=(mean, maximum),
        win=(adaptivewindow(nwindows=5, overlap=0.4),)
        )),
    dataframe=true
)
test4 = get_dataset(
    dt,
    TreatmentGroup(aggrfunc=reducesize(
        win=(adaptivewindow(nwindows=5, overlap=0.4),)
        )),
    dataframe=true
)
