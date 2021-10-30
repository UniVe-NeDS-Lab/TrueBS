from numba import cuda, types
import math
import numpy as np

threads_n = 256
EARTH_RADIUS=6371
EARTH_RADIUS_SQUARED = 40589641
CORNER_OVERLAP = 16

from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

@cuda.jit()
def logical_or_axis_k(viewsheds, output, idx):
    j = cuda.grid(1)
    if j < output.shape[0]:
        for i in range(viewsheds.shape[1]):
            if viewsheds[j,i] > 0:
                output[j, idx] += 1
                break

@cuda.jit()
def take_values(viewshed, positions, intervisibility_mat, index):
    i = cuda.grid(1)
    if i < positions.shape[0] and i != index:
        x, y = positions[i]
        intervisibility_mat[index, i] = viewshed[x,y]

@cuda.jit()
def memset_k(array, val):
    if len(array.shape) >= 2:
        #2d
        i, j = cuda.grid(2)
        if i < array.shape[0] and j < array.shape[1]:
            array[i, j] = val
    else:
        #1d
        i = cuda.grid(1)
        if i < array.shape[0]:
            array[i] = val

@cuda.jit
def sum_k(viewshed, cumulative):
    i, j = cuda.grid(2)
    if i < cumulative.shape[0] and j < cumulative.shape[1]:
        cumulative[i, j] += viewshed[i, j]
        viewshed[i, j] = 0

@cuda.jit
def bool_k(cumulative, boolean):
    i, j = cuda.grid(2)
    if i < cumulative.shape[0] and j < cumulative.shape[1]:
        if cumulative[i, j] > 0:
            boolean[i, j] += 1
        cumulative[i, j] = 0

@cuda.jit()
def viewshed_k(dsm, out, poi, max_dist,  width_resol, height_resol, poi_elev, tgt_elev, poi_elev_type):
    p_offset = cuda.shared.array(shape=(threads_n), dtype=types.float32) #TODO: fix data type
    p_id = cuda.shared.array(shape=(threads_n, 2), dtype=types.int32)
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x - cuda.blockIdx.x * 2;
    p_offset[cuda.threadIdx.x] = 1
    p_id[cuda.threadIdx.x] = (-1,-1)
    x_step = np.float32(1)
    y_step = np.float32(1)
    # INITIALIZE DATA WRT to DIRECTION OF TRACING
    if(cuda.blockIdx.y==0):
        # NORTH
        if(tid>=dsm.shape[0]+CORNER_OVERLAP):
            return
        x_end= tid - CORNER_OVERLAP/2.0
        y_end = np.float32(0)
        xs = x_end - poi[0]
        ys = y_end - poi[1]
        x_step = abs(xs/ys)
        if x_step > 1.0:
            x_step = np.float32(1)
            y_step = ys/xs
        north_south=True
    elif(cuda.blockIdx.y==1):
        # SOUTH
        if(tid>=dsm.shape[0]+CORNER_OVERLAP):
            return
        x_end= tid - CORNER_OVERLAP/2.0
        y_end = dsm.shape[1]

        xs = x_end - poi[0]
        ys = y_end - poi[1]
        x_step = abs(xs/ys)
        if x_step > 1.0:
            x_step = np.float32(1)
            y_step = ys/xs
        north_south=True
    elif(cuda.blockIdx.y==2):
        # WEST
        if(tid>=dsm.shape[1]+CORNER_OVERLAP):
            return
        x_end = np.float32(0)
        y_end = tid - CORNER_OVERLAP/2.0
        xs = x_end - poi[0]
        ys = y_end - poi[1]
        y_step = abs(ys/xs)
        if y_step > 1.0:
            y_step = np.float32(1)
            x_step = xs/ys
        north_south=False
    elif(cuda.blockIdx.y==3):
        # EAST
        if(tid>=dsm.shape[1]+CORNER_OVERLAP):
            return
        x_end = dsm.shape[0]
        y_end = tid - CORNER_OVERLAP/2.0
        xs = x_end - poi[0]
        ys = y_end - poi[1]
        y_step = abs(ys/xs)
        if y_step > 1.0:
            y_step = np.float32(1)
            x_step = xs/ys
        north_south=False
    max_tilt = np.float32(np.inf)
    if(x_step<0.0):
        x_step = -x_step;
    if(xs<0.0):
        x_step = -x_step;

    if(y_step<0.0):
        y_step = -y_step;
    if(ys<0.0):
        y_step = -y_step;

    x=np.float32(0)
    y=np.float32(0)
    colsd = dsm.shape[0]
    rowsd = dsm.shape[1]
    if poi_elev_type == 0:
        #normal old type (origin point + elev)
        h0 = dsm[poi[0],poi[1]] + poi_elev
    elif poi_elev_type == 1:
        #use this to specify the height as third column in coords
        h0 = poi[2] + poi_elev
    step = -1;
    while(True):
        step+=1
        x = x_step * step + poi[0] + 0.5
        y = y_step * step + poi[1] + 0.5
        #Exit conditions
        if(x<0.5):
            break
        if((x+0.5)>colsd):
            break
        if(y<0.5):
            break
        if((y+0.5)>rowsd):
            break
        
        #distance from poi
        xd = x-poi[0]
        yd = y-poi[1]

        #xd *= width_resolution
        #yd *= height_resolution #useless mult by 1

        #2-d distance
        distance = math.sqrt(xd*xd + yd*yd)

        #take neighbor point for closest trace interpolation (see section 4.5 of paper)
        p1 = (int(x),int(y))
        
        if distance < 0.01:
            out[p1] = 1
            continue #too short distance
        if distance > max_dist:
           return

        if(north_south):
            offset = x-int(x)
        else:
            offset = y-int(y)

        p_offset[cuda.threadIdx.x] = offset
        p_id[cuda.threadIdx.x] = p1
        cuda.syncthreads()
        point_consider = 1
        # If the left or right sweep is more precise do not consider this one
        if (((cuda.threadIdx.x<(cuda.blockDim.x-1)) and
             (cuda.threadIdx.x!=0) ) or tid==0):
            if(cuda.threadIdx.x != cuda.blockDim.x):
                if((p_id[cuda.threadIdx.x+1, 0] == p1[0]) and
                   (p_id[cuda.threadIdx.x+1, 1] == p1[1]) and
                   (p_offset[cuda.threadIdx.x+1] <= offset)):
                    point_consider = 0
            if(cuda.threadIdx.x != 0):
                if((p_id[cuda.threadIdx.x-1, 0] == p1[0]) and
                   (p_id[cuda.threadIdx.x-1, 1] == p1[1]) and
                   (p_offset[cuda.threadIdx.x-1] < offset)):
                    point_consider = 0

        small_dist = distance/1000
        h_corr = (math.sqrt(small_dist*small_dist + EARTH_RADIUS_SQUARED) - EARTH_RADIUS)*1000

        h1 = dsm[p1] - h_corr
        h_diff = h0 - h1
        tilt_land = (h_diff)/ distance
        tilt_ant =  (h_diff - tgt_elev)/ distance
        if( tilt_ant <= max_tilt):
            if point_consider:
                out[p1] = 1
        if(tilt_land < max_tilt):
          max_tilt = tilt_land

@cuda.jit()
def parallel_viewshed_t_k(dsm, out, translation_matrix, coords, max_dist,  width_resol, height_resol, poi_elev, tgt_elev, poi_elev_type):
    p_offset = cuda.shared.array(shape=(threads_n), dtype=types.float32) #TODO: fix data type
    p_id = cuda.shared.array(shape=(threads_n, 2), dtype=types.int32)
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x - cuda.blockIdx.x * 2;
    p_offset[cuda.threadIdx.x] = 1
    p_id[cuda.threadIdx.x] = (-1,-1)
    tidz = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
    if tidz >= out.shape[1]:
        return
    poi = coords[tidz]
    x_step = np.float32(1)
    y_step = np.float32(1)
    # INITIALIZE DATA WRT to DIRECTION OF TRACING
    if(cuda.blockIdx.y==0):
        # NORTH
        if(tid>=dsm.shape[0]+CORNER_OVERLAP):
            return
        x_end= tid - CORNER_OVERLAP/2.0
        y_end = np.float32(0)
        xs = x_end - poi[0]
        ys = y_end - poi[1]
        x_step = abs(xs/ys)
        if x_step > 1.0:
            x_step = np.float32(1)
            y_step = ys/xs
        north_south=True
    elif(cuda.blockIdx.y==1):
        # SOUTH
        if(tid>=dsm.shape[0]+CORNER_OVERLAP):
            return
        x_end= tid - CORNER_OVERLAP/2.0
        y_end = dsm.shape[1]

        xs = x_end - poi[0]
        ys = y_end - poi[1]
        x_step = abs(xs/ys)
        if x_step > 1.0:
            x_step = np.float32(1)
            y_step = ys/xs
        north_south=True
    elif(cuda.blockIdx.y==2):
        # WEST
        if(tid>=dsm.shape[1]+CORNER_OVERLAP):
            return
        x_end = np.float32(0)
        y_end = tid - CORNER_OVERLAP/2.0
        xs = x_end - poi[0]
        ys = y_end - poi[1]
        y_step = abs(ys/xs)
        if y_step > 1.0:
            y_step = np.float32(1)
            x_step = xs/ys
        north_south=False
    elif(cuda.blockIdx.y==3):
        # EAST
        if(tid>=dsm.shape[1]+CORNER_OVERLAP):
            return
        x_end = dsm.shape[0]
        y_end = tid - CORNER_OVERLAP/2.0
        xs = x_end - poi[0]
        ys = y_end - poi[1]
        y_step = abs(ys/xs)
        if y_step > 1.0:
            y_step = np.float32(1)
            x_step = xs/ys
        north_south=False
    max_tilt = np.float32(np.inf)
    if(x_step<0.0):
        x_step = -x_step;
    if(xs<0.0):
        x_step = -x_step;

    if(y_step<0.0):
        y_step = -y_step;
    if(ys<0.0):
        y_step = -y_step;

    x=np.float32(0)
    y=np.float32(0)
    colsd = dsm.shape[0]
    rowsd = dsm.shape[1]
    if poi_elev_type == 0:
        #normal old type (origin point + elev)
        h0 = dsm[poi[0],poi[1]] + poi_elev
    elif poi_elev_type == 1:
        #use this to specify the height as third column in coords
        h0 = poi[2] + poi_elev
    step = -1;
    while(True):
        step+=1
        x = x_step * step + poi[0] + 0.5
        y = y_step * step + poi[1] + 0.5
        #Exit conditions
        if(x<0.5):
            break
        if((x+0.5)>colsd):
            break
        if(y<0.5):
            break
        if((y+0.5)>rowsd):
            break

        #distance from poi
        xd = x-poi[0]
        yd = y-poi[1]

        #xd *= width_resolution
        #yd *= height_resolution #useless mult by 1

        #2-d distance
        distance = math.sqrt(xd*xd + yd*yd)

        #take neighbor point for closest trace interpolation (see section 4.5 of paper)
        p1 = (int(x),int(y)) #, tidz)
        if distance < 0.01:
            #out[translation_matrix[p1][0], translation_matrix[p1][1], tidz] = 1
            continue #too short distance
        if distance > max_dist:
           return

        if(north_south):
            offset = x-int(x)
        else:
            offset = y-int(y)

        p_offset[cuda.threadIdx.x] = offset
        p_id[cuda.threadIdx.x] = (p1[0], p1[1])
        cuda.syncthreads()
        point_consider = 1
        # If the left or right sweep is more precise do not consider this one
        if (((cuda.threadIdx.x<(cuda.blockDim.x-1)) and
             (cuda.threadIdx.x!=0) ) or tid==0):
            if(cuda.threadIdx.x != cuda.blockDim.x):
                if((p_id[cuda.threadIdx.x+1, 0] == p1[0]) and
                   (p_id[cuda.threadIdx.x+1, 1] == p1[1]) and
                   (p_offset[cuda.threadIdx.x+1] <= offset)):
                    point_consider = 0
            if(cuda.threadIdx.x != 0):
                if((p_id[cuda.threadIdx.x-1, 0] == p1[0]) and
                   (p_id[cuda.threadIdx.x-1, 1] == p1[1]) and
                   (p_offset[cuda.threadIdx.x-1] < offset)):
                    point_consider = 0

        small_dist = distance/1000
        h_corr = (math.sqrt(small_dist*small_dist + EARTH_RADIUS_SQUARED) - EARTH_RADIUS)*1000
        
        h1 = dsm[p1[0], p1[1]] - h_corr
        h_diff = h0 - h1
        tilt_land = (h_diff)/ distance
        tilt_ant =  (h_diff - tgt_elev)/ distance
        translated_id =  translation_matrix[p1]
        
        if(tilt_ant <= max_tilt):
            if point_consider:
                if translated_id > 0:
                    out[translated_id-1, tidz] = 1
        if(tilt_land < max_tilt):
          max_tilt = tilt_land
