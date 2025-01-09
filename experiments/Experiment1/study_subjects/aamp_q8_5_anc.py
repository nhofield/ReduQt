input_space = 'zero'

def run(qc):
    qc.initialize([(-0.004747421016267061+5.652181436390612e-17j), (-0.0047474210162670154+4.308638775898296e-17j), (-0.004747421016267058+3.665937015411011e-17j), (-0.004747421016267034+2.965096115405969e-17j), (0.27699043711274784+3.3568482082292864e-16j), (-0.0047474210162670475+4.250499636380526e-17j), (-0.0047474210162670345+4.25049963638053e-17j), (-0.004747421016267068+2.2642552154009215e-17j), (-0.004747421016267064+5.594042296872842e-17j), (0.27699043711274784+4.538341230756643e-16j), (-0.004747421016267036+2.965096115405966e-17j), (-0.004747421016267054+2.9069569758881984e-17j), (-0.004747421016267093+4.2504996363805216e-17j), (0.27699043711274784+5.158315873068798e-16j), (-0.004747421016267017+2.2642552154009236e-17j), (-0.004747421016267093+2.8488178363704333e-17j), (0.2769904371127478+3.6376073983368873e-16j), (-0.004747421016267025+3.607797875893249e-17j), (-0.004747421016267084+4.2504996363805253e-17j), (-0.0047474210162671325+2.9069569758881984e-17j), (-0.004747421016267065+4.2504996363805296e-17j), (-0.004747421016267054+3.549658736375482e-17j), (-0.004747421016267058+2.264255215400918e-17j), (-0.004747421016267099+2.20611607588315e-17j), (0.27699043711274784+4.2575820406490427e-16j), (-0.004747421016267067+4.192360496862757e-17j), (-0.004747421016267042+2.906956975888199e-17j), (-0.004747421016267123+2.2061160758831438e-17j), (-0.004747421016267137+3.5496587363754726e-17j), (0.27699043711274784+5.778290515380953e-16j), (-0.004747421016267115+2.206116075883143e-17j), (0.27699043711274784+6.117505967585506e-16j), (-0.004747421016267062+4.30863877589829e-17j), (-0.004747421016267034+3.607797875893252e-17j), (-0.004747421016267087+3.371013016766348e-17j), (-0.004747421016267062+2.9069569758882126e-17j), (-0.004747421016267054+4.2504996363805266e-17j), (-0.0047474210162670345+2.9069569758882064e-17j), (0.27699043711274784+4.596797492853596e-16j), (-0.00474742101626708+2.2061160758831518e-17j), (0.27699043711274784+4.257582040649042e-16j), (0.27699043711274784+5.158315873068799e-16j), (-0.004747421016267061+2.9069569758882015e-17j), (-0.004747421016267089+2.206116075883147e-17j), (-0.0047474210162670545+2.0274703562740286e-17j), (-0.0047474210162670545+1.9693312167562725e-17j), (-0.004747421016267087+2.2061160758831512e-17j), (-0.004747421016267135-1.6913204223350562e-19j), (-0.004747421016267103+4.250499636380518e-17j), (-0.004747421016267064+2.6701721167613074e-17j), (0.27699043711274784+4.877556682961197e-16j), (0.27699043711274784+5.497531325273352e-16j), (-0.004747421016267107+2.0274703562740234e-17j), (0.2769904371127479+5.21677213516575e-16j), (-0.004747421016267156+2.8488178363704185e-17j), (-0.004747421016267158+2.1479769363653762e-17j), (-0.004747421016267123+3.5496587363754806e-17j), (-0.004747421016267078+1.3266294562689874e-17j), (-0.0047474210162671265+2.2061160758831444e-17j), (-0.004747421016267148+1.5052751758781018e-17j), (-0.0047474210162671195+1.5634143153958665e-17j), (-0.004747421016267104+2.1479769363653855e-17j), (-0.0047474210162671065+8.625734153908175e-18j), (-0.004747421016267182+8.044342758730453e-18j), (-0.004747421016267062+3.665937015411011e-17j), (-0.004747421016267079+2.9650961154059653e-17j), (-0.004747421016267073+2.9650961154059665e-17j), (-0.004747421016267062+2.2642552154009205e-17j), (-0.004747421016267045+3.607797875893245e-17j), (-0.004747421016267076+2.2642552154009212e-17j), (-0.004747421016267075+2.2642552154009156e-17j), (-0.004747421016267099+1.563414315395872e-17j), (-0.004747421016267124+2.9650961154059616e-17j), (-0.004747421016267045+2.906956975888198e-17j), (-0.004747421016267042+2.264255215400921e-17j), (-0.004747421016267119+1.5634143153958635e-17j), (-0.004747421016267059+2.2642552154009202e-17j), (-0.004747421016267045+2.206116075883143e-17j), (-0.004747421016267091+1.5634143153958727e-17j), (-0.004747421016267134+8.6257341539082e-18j), (-0.004747421016267017+3.60779787589325e-17j), (-0.004747421016267109+2.264255215400913e-17j), (-0.004747421016267045+2.2642552154009212e-17j), (-0.004747421016267114+1.5634143153958693e-17j), (-0.004747421016267066+2.2642552154009202e-17j), (-0.004747421016267094+1.5634143153958665e-17j), (-0.0047474210162671004+1.5634143153958693e-17j), (-0.004747421016267148+8.62573415390813e-18j), (-0.0047474210162671004+2.9069569758882e-17j), (-0.00474742101626704+1.5634143153958807e-17j), (-0.0047474210162670675+1.5634143153958745e-17j), (-0.004747421016267138+8.62573415390818e-18j), (-0.004747421016267094+1.5634143153958702e-17j), (-0.004747421016267156+1.5052751758781033e-17j), (-0.004747421016267121+8.625734153908159e-18j), (-0.004747421016267128+8.044342758730536e-18j), (-0.004747421016267063+2.9650961154059653e-17j), (-0.004747421016267069+2.2642552154009218e-17j), (-0.0047474210162670736+2.264255215400918e-17j), (-0.004747421016267119+1.5634143153958696e-17j), (-0.004747421016267094+2.2642552154009218e-17j), (-0.004747421016267091+1.563414315395875e-17j), (-0.004747421016267128+2.2061160758831333e-17j), (-0.0047474210162671+8.625734153908264e-18j), (-0.004747421016267017+2.9069569758882076e-17j), (-0.004747421016267156+2.206116075883148e-17j), (-0.004747421016267116+1.5634143153958684e-17j), (-0.004747421016267133+8.625734153908202e-18j), (-0.004747421016267099+1.563414315395874e-17j), (-0.004747421016267133+8.625734153908215e-18j), (-0.004747421016267082+8.625734153908242e-18j), (-0.00474742101626715+1.617325153857674e-18j), (-0.004747421016267104+2.264255215400913e-17j), (-0.004747421016267079+1.5634143153958736e-17j), (-0.004747421016267073+2.206116075883153e-17j), (-0.004747421016267073+1.5052751758780935e-17j), (-0.004747421016267053+1.563414315395879e-17j), (-0.0047474210162671004+1.5052751758781033e-17j), (-0.004747421016267143+8.62573415390816e-18j), (-0.0047474210162671395+1.6173251538576884e-18j), (-0.004747421016267163+1.5634143153958585e-17j), (-0.0047474210162671065+8.625734153908261e-18j), (-0.004747421016267141+8.625734153908147e-18j), (-0.004747421016267163+1.6173251538576645e-18j), (-0.004747421016267095+8.625734153908205e-18j), (-0.004747421016267123+1.6173251538576976e-18j), (-0.004747421016267151+1.6173251538576776e-18j), (-0.004747421016267163-5.391083846192847e-18j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j])