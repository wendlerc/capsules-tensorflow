import numpy as np

def reconMashup(inputs, pred, pics_per_line=10):
    assert len(inputs) == len(pred), "need as many predictions as inputs"
    assert 2*(len(inputs))%pics_per_line == 0
    lines = int((2*len(inputs))/pics_per_line)
    h_pic = inputs[0].shape[0]
    w_pic = inputs[0].shape[1]
    h = int(h_pic*lines)
    w = int(w_pic*pics_per_line)
    out = np.zeros((h,w),dtype=np.float32)
    startrow = 0
    endrow = h_pic 
    startcol = 0 
    endcol = pics_per_line
    i = 0
    for l in range(lines):
        if i%2 == 0:
            out[startrow:endrow] = np.hstack(inputs[startcol:endcol])
        else:
            out[startrow:endrow] = np.hstack(pred[startcol:endcol])
            startcol += pics_per_line
            endcol += pics_per_line
        startrow += h_pic
        endrow += h_pic
        i+=1
    return out