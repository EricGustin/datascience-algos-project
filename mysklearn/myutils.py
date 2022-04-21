def grouping(attributes):
    count = {}
    grouping = []
    names = []
    total = 0
    for val in attributes:
        total +=1
        if val not in count:
            count[val] = 1
        else:
            count[val] +=1
    for k, v in count.items():
        val = round(v/total, 2)
        grouping.append(val)
        names.append(k)
        
    return grouping, names
    

