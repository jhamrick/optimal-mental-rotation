
def run(data, results_path, seed):
    np.random.seed(seed)

    results = {}
    for key, df in data.iteritems():
        df = data[key]
        for flipped, fdf in df[df['correct']].groupby('flipped'):
            time = fdf.groupby('modtheta')['time']
            stats = time.apply(util.bootstrap).unstack(1)
            results[key, flipped] = stats
