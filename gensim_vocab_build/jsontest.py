def something(a,b):
    if a > 11:
        print a, b
        return True
    else:
        return False

for a in xrange(10):
    for b in xrange(20):
        print a, b
        if something(a, b):
            # Break the inner loop...
            break
    else:
        # Continue if the inner loop wasn't broken.
        continue
    # Inner loop was broken, break the outer.
    break