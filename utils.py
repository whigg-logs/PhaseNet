"""
多进程数据读取 
不会有全局锁问题，速度比多线程要快
"""
import matplotlib.pyplot as plt 
import numpy as np 
import obspy 
import os 
import datetime 
from scipy import signal 
import queue, threading, multiprocessing
import tqdm 
plt.switch_backend('agg')

def phase_file(file_name="pick.PHASE.NSZONE.MAG3.0818"):
    files = open(file_name, "r")
    phase_file = []
    event = []
    for itr in files.readlines():
        if "#EVENT" in itr:
            phase_file.append(event)
            event = [] 
            event.append([a for a in itr.strip().split(" ") if len(a)>0]) 
        elif "PHASES" in itr:
            event.append([a for a in itr.strip().split(" ") if len(a)>0]) 
    event_dict = {}  
    for itr in phase_file[1:]:
        event_dict[itr[0][1]] = itr[1:]
    return event_dict 

def process(event, file_dir):
    sac_file = obspy.read(file_dir)[0] 
    sac_head = sac_file.stats.sac 

    sac_time = datetime.datetime(
        year=sac_head.nzyear, 
        month=1, 
        day=1, 
        hour=sac_head.nzhour, 
        minute=sac_head.nzmin, 
        second=sac_head.nzsec, 
        microsecond=int(sac_head.nzmsec*1000)) + datetime.timedelta(
            days=float(sac_head.nzjday-1))

    phase_time = datetime.datetime(
        year=int(event[9]), 
        month=int(event[10]), 
        day=int(event[11]), 
        hour=int(event[13]), 
        minute=int(event[14]), 
        second=int(event[15]), 
        microsecond=int(event[16])*1000)   

    interval = phase_time - sac_time 
    sac_delta = sac_head.delta 
    sac_b = sac_head.b 
    n_begin = int((interval.total_seconds()-sac_b)/sac_delta) 
    data = sac_file.data[:50000] 
    data = signal.detrend(data)
    phase_name = event[7] 
    shift = np.random.randint(20000) 
    seg1 = data[:-shift] 
    seg2 = data[-shift:] 
    std1 = np.std(seg1[:200])
    std2 = np.std(seg2[-200:]) 
    ratio = std1 / (std2 + 1e-6) 
    data = np.concatenate([seg2*ratio, seg1], axis=0) 
    n_begin += shift 
    phase_name = event[7]
    plt.clf()
    plt.plot(data/np.max(np.abs(data))) 
    plt.plot([n_begin, n_begin], [-1, 1])
    plt.text(n_begin, 1, phase_name)
    plt.savefig("tempfig/%s.png"%file_dir.split('/')[-1])
    #print(event[7]) 
    #print([(a, b) for a, b in enumerate(event)])
def find_phase(): 
    event = phase_file("goodCtlg.YN.and.SC.ctlg") 
    base_dir = "/data/yzyBeiJingData/rawdata/NSZONE_MAG3_1ST"
    dir_idx = [2, 3, 4, 6, 7, 8]
    file_idx = [2, 3, 4, 5]
    #event_count = 0
    phase_count = 0 
    phase_files = []
    for event_count, itr_event in enumerate(event):
        dir_str = ""
        for itr in dir_idx:
            dir_str += itr_event[0][itr]
        event_dirs = os.path.join(base_dir, dir_str) 
        is_exist = os.path.exists(event_dirs)
        if is_exist != True:
            continue 
        files = os.listdir(event_dirs)
        id_names = files[0].split('.') 
        for itr_phase in itr_event[1:]:
            name_str = []
            for itr in file_idx:
                name_str.append(itr_phase[itr])
            for itr in id_names[-3:]:
                name_str.append(itr)
            file_name = ".".join(name_str) 
            file_dirs = os.path.join(event_dirs, file_name)
            is_exist = os.path.exists(file_dirs)
            if is_exist:
                #process(itr_phase, file_dirs) 
                phase_files.append([itr_phase, file_dirs]) 
                phase_count += 1
        #print("event", event_count, "phase", phase_count)
    np.savez("phase.npz", phase=phase_files)

class DataTool():
    def __init__(self, phase_file="phase.npz", batch_size=32, n_samples=30000, n_thread=10, n_rsp=4, n_span=200, noize=0.2, is_3c=True):
        self.phase_file = self.read_ctlg(is_3c) 
        print("Phase file finished")
        self.n_thread = n_thread 
        self.n_file = len(self.phase_file) 
        self.batch_size = batch_size
        self.queue = multiprocessing.Queue(maxsize=100) 
        self.in_queue = multiprocessing.Queue(maxsize=100) 
        self.out_queue = multiprocessing.Queue(maxsize=100)
        self.batch_queue = multiprocessing.Queue(maxsize=100)
        self.n_current = 0 
        self.n_samples = n_samples 
        self.sequence = np.array([a for a in range(self.n_file)]) 
        np.random.shuffle(self.sequence) 
        self.epoch = 0 
        self.data_thread = [] 
        self.noize = noize
        self.n_rsp = n_rsp
        self.n_span = n_span
        self.is_3c = is_3c
        for itr in range(n_thread):
            if self.is_3c:
                self.data_thread.append(multiprocessing.Process(target=self.process_multithread_3c, args=(self.in_queue, self.out_queue)))
            else:
                self.data_thread.append(multiprocessing.Process(target=self.process_multithread, args=(self.in_queue, self.out_queue)))
        for itr in self.data_thread:
            itr.start() 
        self.input_thread = multiprocessing.Process(target=self.batch_data_input, args=(self.in_queue,)) 
        self.input_thread.start() 
        self.output_thread = multiprocessing.Process(target=self.batch_data_output, args=(self.out_queue, self.batch_queue)) 
        self.output_thread.start()
    def read_ctlg(self, is_3c): 
        if os.path.exists("all_phase.npz")==True:
            return np.load("all_phase.npz", allow_pickle=True)["all_phase"]
        event_dir = "/data/wangwtCUTV20190928/ctlgAIV20190928MAG2/ctlg.NSZONE.0818.MagOver2.DIREVID.V20190928" 
        events_file = open(event_dir, "r") 
        events = [[a for a in itr.strip().split(" ") if len(a)>1] for itr in events_file.readlines() if len(itr)>0]
        phase_dir = "/data/yzyBeiJingData/catalog/mkphaseNSZONE0818/pick.PHASE.NSZONE.MAG2.0818" 
        phase_files = phase_file(phase_dir) 
        event_dir_base = "/data/wangwtCUTV20190928/NSZONE_MAG2_EVID_MAGLESS3" 
        replace = ["BHE", "BHN", "BHZ"] 
        all_phase = []
        for itr in tqdm.tqdm(events):
            event_dir = os.path.join(event_dir_base, itr[-2]) 
            try:
                event_phase = phase_files[itr[-1]]  
                if os.path.exists(event_dir)==False:continue
                files = os.listdir(event_dir) 
                if len(files) == 0:continue 
                apd = files[0].split('.')[-3:] 
                for it_phase in event_phase:
                    name = ".".join(it_phase[2:6]+apd) 
                    component_name = it_phase[5]
                    sac_dir1 = os.path.join(event_dir, name) 
                    if is_3c:
                        sac_dir = [] 
                        for it_comp_name in replace:
                            if os.path.exists(sac_dir1.replace(component_name, it_comp_name))==False:continue 
                            sac_dir.append(sac_dir1.replace(component_name, it_comp_name)) 
                        if len(sac_dir)<3:continue 
                        all_phase.append([it_phase, sac_dir]) 
                    else:
                        sac_dir = [] 
                        for it_comp_name in replace:
                            if os.path.exists(sac_dir1.replace(component_name, it_comp_name))==False:continue 
                            sac_dir.append(sac_dir1.replace(component_name, it_comp_name)) 
                        if len(sac_dir)<3:continue 
                        all_phase.append([it_phase, sac_dir]) 
            except:
                continue  
                #print(all_phase[0]) 
        np.savez("all_phase.npz", all_phase=all_phase) 
        return all_phase 

        # print(events[0], phase_files[events[0][-1]])
    def process_multithread(self, in_queue, out_queue):
        while True:
            event, file_dir = in_queue.get() 
            sac_file = obspy.read(file_dir[-1])[0] 

            sac_head = sac_file.stats.sac 
            sac_time = datetime.datetime(
                year=sac_head.nzyear, 
                month=1, 
                day=1, 
                hour=sac_head.nzhour , 
                minute=sac_head.nzmin , 
                second=sac_head.nzsec , 
                microsecond=int(sac_head.nzmsec*1000)) + datetime.timedelta(
                    days=float(sac_head.nzjday-1))
            phase_time = datetime.datetime(
                year=int(event[9]), 
                month=int(event[10]), 
                day=int(event[11]), 
                hour=int(event[13]), 
                minute=int(event[14]), 
                second=int(event[15]), 
                microsecond=int(event[16])*1000)   
            interval = phase_time - sac_time 
            sac_delta = sac_head.delta 
            sac_b = sac_head.b 
            n_begin = int((interval.total_seconds()-sac_b)/sac_delta) 
            data = sac_file.data 
            data1 = sac_file1.data 
            data2 = sac_file2.data 
            shift = np.random.randint(n_begin-10000+(self.n_span+100), n_begin-self.n_span-100) 
            #print("Shift", shift)
            data = sac_file.data[shift:shift+self.n_samples]
            data1 = sac_file1.data[shift:shift+self.n_samples]
            data2 = sac_file2.data[shift:shift+self.n_samples] 
            data /= (np.std(data)+1e-6)
            data1 /= (np.std(data1)+1e-6)
            data2 /= (np.std(data2)+1e-6)
            phase_name = event[7] 
            logit_p = np.zeros_like(data) 
            logit_s = np.zeros_like(data) 
            regre_p = np.zeros_like(data) 
            regre_s = np.zeros_like(data) 
            data = data + np.random.normal(0, np.random.random()*self.noize, np.shape(data))

            
            w_lgp = np.zeros_like(data) * 0.0
            w_lgs = np.zeros_like(data) * 0.0
            w_rgp = np.zeros_like(data) * 0.0
            w_rgs = np.zeros_like(data) * 0.0
            span = self.n_span  
            if "P" in phase_name:
                logit_p[n_begin-span:n_begin+span] = 1 
                regre_p[n_begin-span:n_begin+span] = np.linspace(1, -1, span*2) 
                w_lgp[:] = 0.1
                w_lgp[n_begin-span:n_begin+span] = 10  
                w_rgp[n_begin-span:n_begin+span] = 1
            if "S" in phase_name:
                logit_s[n_begin-span:n_begin+span] = 1 
                regre_s[n_begin-span:n_begin+span] = np.linspace(1, -1, span*2)  
                w_lgs[:] = 0.1
                w_lgs[n_begin-span:n_begin+span] = 10 
                w_rgs[n_begin-span:n_begin+span] = 1   
            rsp = self.n_rsp
            out_queue.put([data[::rsp], logit_p[::rsp], 
            logit_s[::rsp], regre_p[::rsp],
            regre_s[::rsp], w_lgp[::rsp], 
            w_lgs[::rsp], w_rgp[::rsp], 
            w_rgs[::rsp], n_begin//rsp]) 
    def process_multithread_3c(self, in_queue, out_queue): 
        while True:
            event, file_dir = in_queue.get()
            sac_file = obspy.read(file_dir[0])[0] 
            sac_file1 = obspy.read(file_dir[1])[0]
            sac_file2 = obspy.read(file_dir[2])[0]
            sac_head = sac_file.stats.sac 
            sac_time = datetime.datetime(
                year=sac_head.nzyear, 
                month=1, 
                day=1, 
                hour=sac_head.nzhour , 
                minute=sac_head.nzmin , 
                second=sac_head.nzsec , 
                microsecond=int(sac_head.nzmsec*1000)) + datetime.timedelta(
                    days=float(sac_head.nzjday-1))
            phase_time = datetime.datetime(
                year=int(event[9]), 
                month=int(event[10]), 
                day=int(event[11]), 
                hour=int(event[13]), 
                minute=int(event[14]), 
                second=int(event[15]), 
                microsecond=int(event[16])*1000)   
            interval = phase_time - sac_time 
            sac_delta = sac_head.delta 
            sac_b = sac_head.b 
            n_begin = int((interval.total_seconds()-sac_b)/sac_delta) 
            data = sac_file.data 
            data1 = sac_file1.data 
            data2 = sac_file2.data 
            shift = np.random.randint(n_begin-10000+(self.n_span+100), n_begin-self.n_span-100) 
            #print("Shift", shift)
            data = sac_file.data[shift:shift+30000]
            data1 = sac_file1.data[shift:shift+30000]
            data2 = sac_file2.data[shift:shift+30000] 
            data /= (np.std(data)+1e-6)
            data1 /= (np.std(data1)+1e-6)
            data2 /= (np.std(data2)+1e-6)
            
            logit_p = np.zeros_like(data) 
            logit_s = np.zeros_like(data) 
            regre_p = np.zeros_like(data) 
            regre_s = np.zeros_like(data) 
            #data = data + np.random.normal(0, np.random.random()*self.noize, np.shape(data))
            n_begin = n_begin-shift
            w_lgp = np.ones_like(data) * 0.1
            w_lgs = np.ones_like(data) * 0.1
            w_rgp = np.zeros_like(data) * 0.0
            w_rgs = np.zeros_like(data) * 0.0
            span = self.n_span 
            phase_name = event[7] 
            #print("Shift", shift, np.shape(data), n_begin, shift, n_begin-span, n_begin+span)
            if "P" in phase_name:
                logit_p[n_begin-span:n_begin+span] = 1 
                regre_p[n_begin-span:n_begin+span] = np.linspace(1, -1, span*2) 
                w_lgp[n_begin-span:n_begin+span] = 5 
                w_rgp[n_begin-span:n_begin+span] = 1
            if "S" in phase_name:
                logit_s[n_begin-span:n_begin+span] = 1 
                regre_s[n_begin-span:n_begin+span] = np.linspace(1, -1, span*2)  
                w_lgs[n_begin-span:n_begin+span] = 5 
                w_rgs[n_begin-span:n_begin+span] = 1   
            rsp = self.n_rsp 
            data = np.vstack([data, data1, data2]).T

            out_queue.put([data[::rsp], logit_p[::rsp], 
            logit_s[::rsp], regre_p[::rsp],
            regre_s[::rsp], w_lgp[::rsp], 
            w_lgs[::rsp], w_rgp[::rsp], 
            w_rgs[::rsp], n_begin//rsp])
    def batch_data_input(self, in_queue):
        while True:
            self.epoch += 1
            for itr in self.sequence:
                in_queue.put(self.phase_file[itr])
    def batch_data_output(self, out_queue, batch_queue):
        while True:
            dt, lgp, lgs, rgp, rgs, wlp, wls, wrp, wrs, tm = [], [], [], [], [], [], [], [], [], []
            for itr in range(self.batch_size):
                d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = out_queue.get()  
                dt.append(d1) 
                lgp.append(d2) 
                lgs.append(d3) 
                rgp.append(d4) 
                rgs.append(d5) 
                wlp.append(d6) 
                wls.append(d7) 
                wrp.append(d8) 
                wrs.append(d9) 
                tm.append(d10)
            lgp = np.vstack(lgp) 
            lgs = np.vstack(lgs) 
            rgp = np.vstack(rgp) 
            rgs = np.vstack(rgs) 
            wlp = np.vstack(wlp) 
            wls = np.vstack(wls) 
            wrp = np.vstack(wrp) 
            wrs = np.vstack(wrs) 
            rgp = np.expand_dims(rgp, 2)
            rgs = np.expand_dims(rgs, 2)
            
            if self.is_3c == False:
                dt = np.vstack(dt)
                dt = np.expand_dims(dt, 2)   
            else:
                dt = np.concatenate([np.expand_dims(a, 0) for a in dt], 0)
            batch_queue.put([dt, lgp, lgs, rgp, rgs, wlp, wls, wrp, wrs])    
    def batch_data(self):
        dt, lgp, lgs, rgp, rgs, wlp, wls, wrp, wrs = self.batch_queue.get() 
        return dt, lgp, lgs, rgp, rgs, wlp, wls, wrp, wrs 
    def close_thread(self):
        #for itr in self.data_thread:
        #    itr.terminate() 
        #self.input_thread.join() 
        #self.output_thread.terminate() 
        pass

if __name__ == "__main__":
    tool = DataTool() 
    dt, lgp, lgs, rgp, rgs, wlp, wls, wrp, wrs = tool.batch_data() 
    for itr in range(32):
        plt.clf() 
        plt.subplot(311)
        plt.plot(dt[itr, :, 0]) 
        plt.subplot(312)
        plt.plot(lgp[itr, :], c="r")
        plt.plot(lgs[itr, :], c="b") 
        plt.subplot(313)
        plt.plot(rgp[itr, :], c="r")
        plt.plot(rgs[itr, :], c="b")         
        plt.savefig("tempfig/t%d.png"%itr)
    #tool.read_ctlg()