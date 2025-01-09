input_space = 'zero'

def run(qc):
    qc.initialize([(0.11328124999999997+4.729959072878694e-17j), (-0.04296874999999992-4.765837397072449e-17j), (0.11328124999999996+1.9511828640711966e-16j), (0.11328124999999994+2.0899123842871078e-16j), (-0.04296874999999992-4.765837397072446e-17j), (-0.042968749999999944-5.292052818581077e-17j), (0.11328125+1.9176964281570122e-16j), (-0.04296874999999997-6.009619302456487e-17j), (0.11328124999999997+1.778966907941101e-16j), (-0.042968749999999944-5.292052818581082e-17j), (-0.04296874999999993-6.057457068048175e-17j), (0.11328124999999999+2.22864190450302e-16j), (-0.04296874999999997-5.2920528185810806e-17j), (-0.04296874999999996-6.392321427190029e-17j), (0.11328124999999997+2.22864190450302e-16j), (-0.042968749999999986-6.72718578633189e-17j), (0.11328124999999994+1.7789669079411006e-16j), (-0.0429687499999999-5.2920528185810775e-17j), (-0.04296874999999996-5.2920528185810794e-17j), (0.11328124999999996+1.903345098479504e-16j), (-0.04296874999999992-5.292052818581077e-17j), (0.11328125+2.22864190450302e-16j), (-0.04296874999999992-5.818268240089706e-17j), (0.11328124999999999+2.7118033369791234e-16j), (0.11328124999999992+1.4201836660033997e-16j), (-0.04296874999999995-6.200970364823259e-17j), (-0.042968749999999944-5.818268240089706e-17j), (0.11328124999999992+2.5395873808490276e-16j), (-0.04296874999999996-7.349076739023898e-17j), (-0.04296874999999996-6.918536848698654e-17j), (-0.04296874999999999-8.066643222899311e-17j), (-0.04296875000000005-7.4447522702073e-17j), (-0.0429687499999999-4.3831352723389e-17j), (-0.042968749999999924-5.4834038809478524e-17j), (-0.042968749999999924-5.483403880947847e-17j), (-0.04296874999999993-6.39232142719003e-17j), (-0.04296874999999993-6.631510255148496e-17j), (0.11328124999999999+2.0564259483729234e-16j), (-0.04296874999999995-6.200970364823252e-17j), (-0.04296874999999999-7.683941098165755e-17j), (0.11328124999999997+2.0899123842871085e-16j), (-0.042968749999999944-6.200970364823253e-17j), (0.11328125+2.4008578606331167e-16j), (-0.042968750000000014-6.535834723965114e-17j), (-0.042968749999999944-6.200970364823256e-17j), (0.11328124999999994+2.7118033369791234e-16j), (-0.042968749999999965-7.683941098165751e-17j), (0.11328124999999997+2.353020095041423e-16j), (0.11328125000000001+1.9176964281570127e-16j), (-0.042968749999999944-6.39232142719003e-17j), (-0.04296874999999994-6.00961930245648e-17j), (-0.04296874999999997-8.066643222899307e-17j), (-0.042968749999999924-6.200970364823256e-17j), (-0.042968749999999965-6.53583472396511e-17j), (0.11328124999999994+2.042074618695415e-16j), (0.11328125+2.3530200950414233e-16j), (-0.04296874999999996-5.626917177722936e-17j), (-0.042968749999999965-6.918536848698664e-17j), (0.11328125+2.367371424718932e-16j), (-0.04296875000000003-7.636103332574075e-17j), (-0.04296874999999999-7.109887911065444e-17j), (0.11328125+2.678316901064939e-16j), (-0.04296875-7.636103332574074e-17j), (-0.04296875000000004-8.162318754082712e-17j), (-0.042968749999999896-5.722592708906321e-17j), (-0.04296874999999993-5.292052818581078e-17j), (0.11328124999999994+1.7454804720269154e-16j), (-0.042968749999999944-6.200970364823256e-17j), (-0.04296874999999991-5.1007017562142995e-17j), (0.11328124999999997+2.0564259483729236e-16j), (-0.04296874999999994-5.81826824008971e-17j), (-0.04296874999999997-6.918536848698669e-17j), (-0.042968749999999924-6.248808130414946e-17j), (0.11328124999999997+1.8842099922428268e-16j), (-0.04296874999999994-5.818268240089707e-17j), (-0.04296875-6.727185786331892e-17j), (0.11328124999999997+1.7311291423494071e-16j), (-0.042968750000000014-6.727185786331898e-17j), (0.11328124999999996+2.8840192931092197e-16j), (-0.042968750000000014-7.444752270207299e-17j), (-0.042968749999999944-5.2920528185810794e-17j), (-0.04296874999999996-6.39232142719003e-17j), (-0.042968749999999944-6.009619302456483e-17j), (-0.042968749999999965-7.109887911065437e-17j), (0.11328125+2.400857860633116e-16j), (0.11328125+2.711803336979124e-16j), (0.11328124999999994+2.711803336979124e-16j), (-0.042968750000000014-7.253401207840525e-17j), (-0.042968749999999965-6.966374614290354e-17j), (-0.042968749999999944-6.91853684869866e-17j), (0.11328125+2.3673714247189313e-16j), (-0.042968750000000035-8.592858644407939e-17j), (0.11328125+2.7118033369791234e-16j), (-0.04296875000000002-8.592858644407947e-17j), (0.11328124999999992+3.367180725585324e-16j), (0.11328124999999996+2.9892623774109457e-16j), (0.11328124999999997+2.262128340417205e-16j), (0.11328124999999994+2.056425948372923e-16j), (-0.04296874999999996-6.00961930245648e-17j), (-0.04296875000000001-6.727185786331895e-17j), (0.11328124999999993+2.0564259483729226e-16j), (-0.04296874999999996-7.492590035798985e-17j), (-0.04296874999999996-6.535834723965116e-17j), (0.11328124999999994+2.678316901064938e-16j), (0.11328124999999996+2.0755610546096002e-16j), (-0.04296874999999996-6.344483661598341e-17j), (0.11328124999999999+2.884019293109221e-16j), (0.11328124999999997+2.8505328571950353e-16j), (-0.042968749999999965-7.683941098165756e-17j), (-0.042968749999999965-7.636103332574068e-17j), (0.11328124999999992+2.525236051171519e-16j), (0.11328124999999997+2.510884721494012e-16j), (0.11328125+1.9033450984795052e-16j), (-0.04296874999999997-6.727185786331893e-17j), (-0.04296875-6.918536848698669e-17j), (-0.042968750000000014-8.018805457307624e-17j), (-0.042968749999999986-6.535834723965114e-17j), (-0.04296874999999996-7.827454394940837e-17j), (0.11328124999999992+2.850532857195035e-16j), (-0.042968750000000014-8.162318754082705e-17j), (0.11328124999999997+2.7118033369791234e-16j), (0.11328124999999994+2.678316901064939e-16j), (-0.042968750000000014-7.253401207840528e-17j), (-0.04296875000000002-8.353669816449483e-17j), (-0.04296875000000004-7.636103332574081e-17j), (-0.042968750000000056-8.353669816449489e-17j), (-0.04296875000000002-8.162318754082704e-17j), (-0.04296875000000006-9.071236300324898e-17j), (-0.0429687499999999-4.9571884594392205e-17j), (-0.04296874999999991-5.866106005681395e-17j), (-0.042968749999999965-5.674754943314624e-17j), (-0.04296874999999992-6.392321427190022e-17j), (-0.042968749999999924-5.866106005681395e-17j), (-0.04296874999999994-6.583672489556803e-17j), (-0.04296874999999993-6.392321427190027e-17j), (-0.042968749999999965-7.301238973432213e-17j), (-0.04296874999999992-5.674754943314623e-17j), (-0.042968749999999965-6.583672489556803e-17j), (-0.04296874999999994-6.583672489556807e-17j), (-0.04296874999999995-7.109887911065433e-17j), (-0.04296874999999994-6.583672489556802e-17j), (-0.042968749999999965-7.301238973432209e-17j), (-0.04296874999999996-7.10988791106544e-17j), (-0.04296874999999998-8.018805457307618e-17j), (-0.042968749999999924-5.674754943314628e-17j), (-0.04296874999999995-6.583672489556803e-17j), (-0.04296874999999994-6.583672489556803e-17j), (-0.042968749999999965-7.109887911065433e-17j), (-0.042968749999999924-6.583672489556798e-17j), (-0.042968749999999986-7.109887911065434e-17j), (-0.042968749999999965-7.301238973432209e-17j), (-0.04296875000000002-7.827454394940846e-17j), (-0.042968749999999944-6.392321427190033e-17j), (-0.04296874999999997-7.301238973432213e-17j), (-0.04296874999999998-7.301238973432214e-17j), (-0.04296874999999996-7.827454394940849e-17j), (-0.042968749999999965-7.301238973432213e-17j), (-0.04296874999999999-8.01880545730762e-17j), (-0.042968750000000014-8.018805457307626e-17j), (-0.04296875000000004-8.736371941183035e-17j), (-0.04296874999999994-5.8661060056814e-17j), (-0.042968749999999944-6.583672489556803e-17j), (-0.042968749999999944-6.583672489556804e-17j), (-0.042968749999999986-7.301238973432213e-17j), (-0.04296874999999995-6.583672489556805e-17j), (-0.042968749999999965-7.109887911065439e-17j), (-0.042968749999999986-7.301238973432218e-17j), (-0.04296875-8.018805457307623e-17j), (-0.042968749999999944-6.392321427190025e-17j), (-0.04296874999999997-7.301238973432212e-17j), (-0.042968749999999986-7.109887911065442e-17j), (-0.04296875000000001-8.018805457307623e-17j), (-0.04296874999999998-7.301238973432209e-17j), (-0.04296875000000001-7.827454394940838e-17j), (-0.042968750000000014-8.018805457307625e-17j), (-0.042968750000000035-8.54502087881626e-17j), (-0.04296874999999997-6.392321427190035e-17j), (-0.042968749999999965-7.301238973432212e-17j), (-0.04296875-7.301238973432213e-17j), (-0.04296875000000001-8.01880545730762e-17j), (-0.04296874999999997-7.301238973432209e-17j), (-0.042968749999999986-8.01880545730762e-17j), (-0.04296874999999999-7.827454394940849e-17j), (-0.042968750000000014-8.545020878816256e-17j), (-0.04296874999999998-7.301238973432208e-17j), (-0.042968749999999986-8.01880545730762e-17j), (-0.042968750000000014-7.827454394940844e-17j), (-0.042968750000000035-8.736371941183031e-17j), (-0.042968750000000014-8.01880545730762e-17j), (-0.04296875000000003-8.545020878816258e-17j), (-0.04296875000000002-8.73637194118303e-17j), (-0.04296875000000007-9.453938425058447e-17j), (-0.04296874999999993-5.8661060056814e-17j), (-0.04296874999999995-6.583672489556807e-17j), (-0.042968749999999986-6.392321427190035e-17j), (-0.04296874999999998-7.301238973432213e-17j), (-0.04296874999999997-6.583672489556807e-17j), (-0.04296874999999997-7.109887911065437e-17j), (-0.042968749999999986-7.301238973432212e-17j), (-0.04296875-8.01880545730762e-17j), (-0.04296874999999996-6.583672489556807e-17j), (-0.04296874999999998-7.10988791106544e-17j), (-0.04296875-7.30123897343222e-17j), (-0.04296874999999999-8.018805457307619e-17j), (-0.04296874999999998-7.109887911065439e-17j), (-0.04296875000000001-8.018805457307619e-17j), (-0.04296874999999998-7.827454394940846e-17j), (-0.042968750000000035-8.736371941183031e-17j), (-0.042968749999999965-6.583672489556805e-17j), (-0.04296874999999999-7.301238973432214e-17j), (-0.042968749999999965-7.301238973432209e-17j), (-0.04296874999999997-8.018805457307619e-17j), (-0.042968749999999986-7.109887911065439e-17j), (-0.042968750000000014-7.827454394940843e-17j), (-0.04296874999999998-7.827454394940843e-17j), (-0.042968750000000035-8.736371941183029e-17j), (-0.04296875-7.301238973432208e-17j), (-0.04296875-8.018805457307624e-17j), (-0.042968750000000014-7.827454394940845e-17j), (-0.042968750000000014-8.736371941183026e-17j), (-0.04296875000000003-7.827454394940849e-17j), (-0.04296875000000003-8.73637194118303e-17j), (-0.04296875000000003-8.545020878816255e-17j), (-0.04296875000000005-9.262587362691667e-17j), (-0.042968749999999944-6.39232142719003e-17j), (-0.042968749999999965-7.109887911065439e-17j), (-0.04296874999999998-7.301238973432213e-17j), (-0.042968750000000014-8.018805457307621e-17j), (-0.042968749999999965-7.10988791106543e-17j), (-0.04296875000000001-8.018805457307619e-17j), (-0.04296875000000002-8.018805457307619e-17j), (-0.04296875000000004-8.545020878816258e-17j), (-0.04296874999999994-7.109887911065429e-17j), (-0.04296875-8.018805457307623e-17j), (-0.04296875000000002-7.827454394940846e-17j), (-0.042968750000000035-8.545020878816256e-17j), (-0.04296875-8.018805457307618e-17j), (-0.04296875000000004-8.736371941183036e-17j), (-0.04296875-8.545020878816251e-17j), (-0.04296875000000006-9.262587362691665e-17j), (-0.04296875-7.109887911065445e-17j), (-0.04296875000000001-8.018805457307621e-17j), (-0.04296875000000005-8.018805457307628e-17j), (-0.042968750000000035-8.736371941183033e-17j), (-0.04296875000000001-8.018805457307623e-17j), (-0.042968750000000035-8.736371941183033e-17j), (-0.04296875000000002-8.545020878816263e-17j), (-0.042968750000000035-9.453938425058442e-17j), (-0.04296874999999998-7.827454394940844e-17j), (-0.04296875000000001-8.54502087881626e-17j), (-0.04296874999999999-8.736371941183029e-17j), (-0.04296875000000007-9.453938425058449e-17j), (-0.042968750000000014-8.736371941183025e-17j), (-0.04296875000000009-9.453938425058446e-17j), (-0.04296875000000007-9.453938425058441e-17j), (-0.0429687500000001-1.0171504908933855e-16j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j])