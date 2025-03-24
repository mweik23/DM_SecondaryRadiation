# -*- python -*-

"""\
A simple pure-Python parser for HepMC IO_GenEvent ASCII event files, which may
be a lot more convenient than using either the native C++ HepMC API, or wrappers
like pyhepmc!
"""

__version__ = "1.0.1"
__author__ = "Andy Buckley <andy@insectnation.org>"


class Particle(object):
    def __init__(self, pid=0, mom=[0,0,0,0], barcode=0, event=None):
        self.evt = event
        self.barcode = barcode
        self.pid = pid
        self.status = None
        self.mom = list(mom)
        self.nvtx_start = None
        self.nvtx_end = None
        self.mass = None

    def vtx_start(self):
        return self.evt.vertices.get(self.nvtx_start) if self.evt else None

    def vtx_end(self):
        return self.evt.vertices.get(self.nvtx_end) if self.evt else None

    def parents(self):
        return self.vtx_start().parents() if self.vtx_start() else None

    def children(self):
        return self.vtx_end().children() if self.vtx_end() else None

    def __repr__(self):
        return "P" + str(self.barcode)
        # return "Particle[bc=%d, st=%d, pid=%d, v0=%d, v1=%d, p=(%1.2e, %1.2e, %1.2e; %1.2e)]" \
        #        % (self.barcode, self.status, self.pid,
        #           self.nvtx_start, self.nvtx_end,
        #           self.mom[0], self.mom[1], self.mom[2], self.mom[3])


class Vertex(object):
    def __init__(self, pos=[0,0,0,0], barcode=0, event=None):
        self.evt = event
        self.pos = list(pos)
        self.barcode = barcode

    def parents(self):
        return [p for p in self.evt.particles.values() if p.nvtx_end == self.barcode]

    def children(self):
        return [p for p in self.evt.particles.values() if p.nvtx_start == self.barcode]

    def __repr__(self):
        return "V" + str(self.barcode)
        # return "Vertex[bc=%d, x=(%1.2e, %1.2e, %1.2e; %1.2e)]" \
        #        % (self.barcode, self.pos[0], self.pos[1], self.pos[2], self.pos[3])


class Event(object):
    def __init__(self):
        self.num = None
        self.weights = None
        self.units = [None, None]
        self.xsec = [None, None]
        self.particles = {}
        self.vertices = {}

    def __repr__(self):
        return "E%d. #p=%d #v=%d, xs=%1.2e+-%1.2e" % \
               (self.num, len(self.particles), len(self.vertices),
                self.xsec[0], self.xsec[1])


class HepMCReader(object):

    def __init__(self, filename):
        self._file = open(filename)
        self._currentline = None
        self._currentvtx = None
        self.version = None
        ## First non-empty line should be the version info
        while True:
            self._read_next_line()
            if self._currentline.startswith("HepMC::Version"):
                self.version = self._currentline.split()[1]
                break
        ## Read on until we see the START_EVENT_LISTING marker
        while True:
            self._read_next_line()
            if self._currentline == "HepMC::IO_GenEvent-START_EVENT_LISTING":
                break
        ## Read one more line to make the first E line current
        self._read_next_line()

    def _read_next_line(self):
        "Return the next line, stripped of the trailing newline"
        self._currentline = self._file.readline()
        if not self._currentline: # no newline means it's the end of the file
            return False
        if self._currentline.endswith("\n"):
            self._currentline = self._currentline[:-1] # strip the newline
        return True

    def next(self):
        "Return a new event graph"
        evt = Event()
        if not self._currentline or self._currentline == "HepMC::IO_GenEvent-END_EVENT_LISTING":
            return None
        assert self._currentline.startswith("E ")
        vals = self._currentline.split()
        evt.num = int(vals[1])
        evt.weights = [float(vals[-1])] # TODO: do this right, and handle weight maps
        ## Read the other event header lines until a Vertex line is encountered
        while not self._currentline.startswith("V "):
            self._read_next_line()
            vals = self._currentline.split()
            if vals[0] == "U":
                evt.units = vals[1:3]
            elif vals[0] == "C":
                evt.xsec = [float(x) for x in vals[1:3]]
        ## Read the event content lines until an Event line is encountered
        while not self._currentline.startswith("E "):
            vals = self._currentline.split()
            if not self._currentline or self._currentline == "HepMC::IO_GenEvent-END_EVENT_LISTING":
                break
            elif vals[0] == "P":
                bc = int(vals[1])
                try:
                    p = Particle(barcode=bc, pid=int(vals[2]), mom=[float(x) for x in vals[3:7]], event=evt)
                    p.mass = float(vals[7])
                    p.status = int(vals[8])
                    p.nvtx_start = self._currentvtx
                    p.nvtx_end = int(vals[11])
                    evt.particles[bc] = p
                except:
                    print(vals)
            elif vals[0] == "V":
                bc = int(vals[1])
                self._currentvtx = bc # current vtx barcode for following Particles
                v = Vertex(barcode=bc, pos=[float(x) for x in vals[3:7]], event=evt)
                evt.vertices[bc] = v
            self._read_next_line()
        return evt


class HepMCWriter(object):

    def __init__(self, filename):
        self._file = open(filename)
        self._file.write("HepMC::Version 2.6.X")
        self._file.write("HepMC::IO_GenEvent-START_EVENT_LISTING\n")

    def finalize(self):
        "Close the HepMC event file"
        self._file.write("HepMC::IO_GenEvent-END_EVENT_LISTING\n")
        self._file.close()
        self._file = None

    # TODO: FINISH!
    def write_next(self, evt):
        "Write the next event"
        self._file.write("E\n")
        self._file.write("U\n") # evt.units
        self._file.write("U\n") # evt.xsec
        for v in evt.vertices:
            self._file.write("V\n")
            for p in v.particles_in + v.particles_out:
                self._file.write("P\n")


def mk_nx_graph(evt):
    "Make a NetworkX graph"
    import networkx as nx
    nxevt = nx.Graph()
    nxevt.add_nodes_from(evt.vertices.values())
    for p in evt.particles.values():
        nxevt.add_edge(evt.vertices[p.nvtx_start],
                       evt.vertices[p.nvtx_end] if p.nvtx_end else p.nvtx_end,
                       object=p)
        # TODO: Why aren't there enough particles?
    return nxevt


if __name__ == "__main__":

    import sys
    mcfile = sys.argv[1] if len(sys.argv) > 1 else "out.hepmc"
    r = HepMCReader(mcfile)
    print("version =", r.version)
    while True:
        evt = r.next()
        if not evt:
            break # end of file
        print(evt)
        # for p in evt.particles: print p
        # for v in evt.vertices: print v

        # ## Make an NX event graph
        # nxevt = mk_nx_graph(evt)
        # print "#p =", nxevt.number_of_nodes(), "#v =", nxevt.number_of_edges()

        # ## Plot the NX event graph
        # import matplotlib.pyplot as plt
        # # nx.draw_circular(nxevt)
        # # nx.draw_spectral(nxevt)
        # # nx.draw_random(nxevt)
        # pos = nx.graphviz_layout(nxevt, prog='dot', args='')
        # plt.figure(figsize=(8,8))
        # nx.draw(nxevt, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
        # plt.savefig("nx-%d.pdf" % evt.num)

        # break
