import bootstrap

def main():
    bs = bootstrap.Bootstrap('rhd.csv')
    conf_interval = bs.getBootstrapCIs(0.05,bs.X.iloc[[0]])
    print('confidence interval:',conf_interval)
    print('actual execution time:', bs.y.iloc[[0]])
    
#     n = len(bs.y)
#     count = 0
#     for i in range(n):
#     	conf_interval = bs.getBootstrapCIs(0.05,bs.X.iloc[[i]])
#     	if conf_interval[0] < bs.y.iloc[[i]].values < conf_interval[1]:
#     		count += 1
#     	#print('confidence interval:',conf_interval)
#     	#print('actual execution time:', bs.y.iloc[[0]])
#     print('Fraction of hits:',count*1./n)

if __name__ == "__main__":
    main()