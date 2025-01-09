input_space = 'zero'

def run(qc):
    qc.initialize([(-0.03997802734374993-4.4784370709786036e-17j), (0.12017822265624999+1.978801699049527e-16j), (-0.03997802734374991-6.068295311815044e-17j), (0.12017822265624996+1.9513697303430403e-16j), (-0.03997802734374995-4.752756758043472e-17j), (0.12017822265624994+2.1259775747527202e-16j), (-0.03997802734374993-5.6728862805962e-17j), (-0.03997802734374995-6.377745857986313e-17j), (-0.03997802734374992-4.9680267032060895e-17j), (0.12017822265625003+1.9513697303430407e-16j), (-0.03997802734374993-5.672886280596194e-17j), (0.12017822265625003+2.4477612948655974e-16j), (-0.03997802734374994-5.457616335433575e-17j), (-0.03997802734374997-7.04747457627002e-17j), (-0.03997802734374993-6.593015803148924e-17j), (-0.03997802734374997-7.297875380539047e-17j), (-0.03997802734374996-5.183296648368716e-17j), (-0.039978027343749965-6.773154889205154e-17j), (-0.03997802734374988-5.4576163354335773e-17j), (-0.03997802734374991-6.59301580314893e-17j), (0.12017822265624994+2.1259775747527212e-16j), (0.12017822265625+2.622369139275278e-16j), (0.12017822265625+2.2731534504559164e-16j), (-0.03997802734374995-6.8673354902138e-17j), (-0.03997802734374992-5.4576163354335786e-17j), (-0.039978027343749944-7.262744521432637e-17j), (-0.03997802734374996-6.37774585798631e-17j), (-0.03997802734374999-7.08260543537643e-17j), (0.12017822265625+2.4477612948655983e-16j), (0.12017822265624999+2.5949371705687915e-16j), (-0.03997802734374995-6.867335490213801e-17j), (-0.03997802734374997-7.787465012766535e-17j), (-0.039978027343749896-4.9680267032060895e-17j), (0.12017822265624992+1.9513697303430403e-16j), (-0.03997802734374988-5.888156225758813e-17j), (-0.03997802734374998-5.94720596766107e-17j), (-0.03997802734374993-5.6728862805962e-17j), (0.12017822265625001+2.6223691392752783e-16j), (-0.03997802734374994-6.593015803148928e-17j), (-0.039978027343750014-7.082605435376429e-17j), (0.12017822265625+1.9513697303430398e-16j), (-0.03997802734374995-5.731936022498453e-17j), (-0.039978027343749944-5.947205967661072e-17j), (0.12017822265624993+2.42032932615911e-16j), (0.12017822265624997+2.4477612948655974e-16j), (0.12017822265624992+2.7695450149784725e-16j), (-0.03997802734374997-6.867335490213799e-17j), (-0.03997802734374997-7.572195067603911e-17j), (-0.03997802734374994-5.672886280596203e-17j), (-0.039978027343749924-7.478014466595258e-17j), (0.12017822265624997+2.0985456060462347e-16j), (-0.039978027343749944-7.297875380539043e-17j), (-0.03997802734374995-6.16247591282369e-17j), (0.12017822265624999+2.594937170568791e-16j), (-0.03997802734374997-6.652065545051182e-17j), (-0.03997802734374998-7.78746501276654e-17j), (0.12017822265624997+2.6223691392752783e-16j), (-0.03997802734374994-7.08260543537642e-17j), (0.12017822265624997+2.7695450149784725e-16j), (-0.039978027343749986-7.572195067603918e-17j), (0.12017822265624992+3.1187607037978344e-16j), (-0.03997802734375-7.141655177278681e-17j), (0.12017822265624993+3.0913287350913476e-16j), (0.12017822265625003+3.0638967663848623e-16j), (-0.0399780273437499-4.9680267032060914e-17j), (-0.03997802734374994-5.888156225758823e-17j), (0.12017822265624997+2.1259775747527212e-16j), (-0.03997802734374998-6.162475912823696e-17j), (0.12017822265624992+1.9513697303430395e-16j), (-0.039978027343749965-6.377745857986311e-17j), (0.12017822265624994+2.7969769836849593e-16j), (-0.03997802734374996-7.967604098822751e-17j), (-0.03997802734374995-5.4576163354335816e-17j), (-0.03997802734374999-6.377745857986318e-17j), (0.12017822265625+2.6223691392752783e-16j), (-0.039978027343749986-6.8673354902138e-17j), (0.12017822265625003+2.2731534504559164e-16j), (-0.039978027343749924-7.082605435376413e-17j), (0.12017822265624997+2.944152859388154e-16j), (-0.039978027343749986-8.002734957929159e-17j), (-0.03997802734374998-5.672886280596203e-17j), (-0.03997802734374999-6.593015803148935e-17j), (0.12017822265624994+2.4477612948655964e-16j), (-0.03997802734375-6.867335490213806e-17j), (-0.03997802734374996-6.162475912823693e-17j), (-0.039978027343749986-7.082605435376423e-17j), (0.12017822265625003+3.1187607037978354e-16j), (0.12017822265625+2.742113046271986e-16j), (0.12017822265625006+2.4477612948655974e-16j), (0.12017822265625003+2.7695450149784735e-16j), (0.12017822265624997+2.9441528593881545e-16j), (0.12017822265625+2.916720890681668e-16j), (-0.03997802734374996-6.652065545051181e-17j), (-0.039978027343750014-7.787465012766538e-17j), (-0.03997802734375003-7.572195067603921e-17j), (-0.03997802734375004-8.492324590156661e-17j), (-0.039978027343749944-6.103426170921441e-17j), (-0.039978027343749965-6.377745857986313e-17j), (-0.03997802734374991-5.947205967661066e-17j), (0.12017822265624994+2.4203293261591106e-16j), (-0.03997802734374995-6.377745857986309e-17j), (-0.039978027343749986-6.867335490213801e-17j), (0.12017822265625+2.944152859388154e-16j), (0.12017822265625001+2.9167208906816677e-16j), (-0.03997802734374996-6.162475912823693e-17j), (-0.03997802734374999-7.082605435376425e-17j), (0.12017822265624997+2.5949371705687925e-16j), (-0.03997802734375005-7.572195067603927e-17j), (-0.03997802734374996-6.436795599888557e-17j), (0.12017822265625+2.742113046271986e-16j), (0.12017822265625+3.0913287350913486e-16j), (-0.03997802734374998-9.37732325360298e-17j), (-0.039978027343749965-6.593015803148935e-17j), (-0.03997802734375-7.297875380539054e-17j), (-0.03997802734374998-7.967604098822758e-17j), (-0.03997802734375001-8.002734957929167e-17j), (-0.03997802734375002-7.082605435376428e-17j), (-0.039978027343750014-8.002734957929163e-17j), (0.12017822265625001+2.916720890681667e-16j), (-0.03997802734375004-8.277054644994035e-17j), (0.12017822265624994+2.944152859388154e-16j), (0.12017822265625+2.9167208906816677e-16j), (-0.039978027343749986-7.3569251224413e-17j), (-0.03997802734375001-8.492324590156654e-17j), (0.12017822265624994+2.916720890681667e-16j), (-0.03997802734375005-8.277054644994033e-17j), (-0.03997802734375004-8.061784699831415e-17j), (-0.03997802734375008-9.412454112709399e-17j), (-0.039978027343749924-5.398566593531332e-17j), (-0.03997802734374988-5.888156225758816e-17j), (-0.0399780273437499-6.103426170921435e-17j), (-0.03997802734374994-6.593015803148928e-17j), (-0.039978027343749944-6.103426170921442e-17j), (-0.03997802734374996-6.593015803148928e-17j), (-0.039978027343749924-6.80828574831155e-17j), (-0.039978027343749944-7.513145325701666e-17j), (-0.03997802734374994-6.103426170921438e-17j), (-0.03997802734374995-6.593015803148927e-17j), (-0.03997802734374994-6.808285748311555e-17j), (-0.03997802734374996-7.297875380539039e-17j), (-0.0399780273437499-6.808285748311548e-17j), (-0.03997802734374995-7.513145325701661e-17j), (-0.03997802734374999-7.513145325701663e-17j), (-0.039978027343749986-8.218004903091782e-17j), (-0.03997802734374993-6.103426170921441e-17j), (-0.039978027343749944-6.80828574831155e-17j), (-0.03997802734374992-6.808285748311546e-17j), (-0.039978027343749944-7.513145325701663e-17j), (-0.039978027343749965-6.593015803148933e-17j), (-0.03997802734374998-7.297875380539042e-17j), (-0.03997802734374996-7.297875380539044e-17j), (-0.03997802734375-8.218004903091773e-17j), (-0.03997802734374995-6.808285748311553e-17j), (-0.039978027343749965-7.513145325701667e-17j), (-0.039978027343749965-7.513145325701666e-17j), (-0.039978027343749986-8.218004903091773e-17j), (-0.03997802734374995-7.297875380539047e-17j), (-0.03997802734374999-8.002734957929153e-17j), (-0.03997802734375-8.21800490309178e-17j), (-0.03997802734375-8.922864480481893e-17j), (-0.03997802734374993-6.103426170921441e-17j), (-0.039978027343749944-6.59301580314893e-17j), (-0.039978027343749944-6.808285748311547e-17j), (-0.039978027343749944-7.513145325701666e-17j), (-0.039978027343749965-6.808285748311555e-17j), (-0.039978027343749965-7.297875380539044e-17j), (-0.03997802734374996-7.513145325701661e-17j), (-0.03997802734375001-8.21800490309178e-17j), (-0.03997802734374998-6.593015803148935e-17j), (-0.039978027343749986-7.513145325701667e-17j), (-0.039978027343749944-7.513145325701665e-17j), (-0.03997802734375002-8.002734957929164e-17j), (-0.03997802734374997-7.297875380539055e-17j), (-0.03997802734375-8.002734957929168e-17j), (-0.03997802734374999-8.21800490309178e-17j), (-0.03997802734375002-8.922864480481895e-17j), (-0.039978027343749944-6.808285748311555e-17j), (-0.03997802734374997-7.513145325701666e-17j), (-0.03997802734374996-7.297875380539045e-17j), (-0.03997802734374999-8.218004903091782e-17j), (-0.039978027343749965-7.513145325701667e-17j), (-0.03997802734374999-8.002734957929162e-17j), (-0.039978027343749986-8.218004903091783e-17j), (-0.039978027343750014-8.922864480481893e-17j), (-0.03997802734374997-7.297875380539052e-17j), (-0.03997802734375-8.218004903091786e-17j), (-0.039978027343749986-8.002734957929152e-17j), (-0.03997802734375002-8.9228644804819e-17j), (-0.039978027343749986-8.002734957929163e-17j), (-0.039978027343750014-8.922864480481896e-17j), (-0.03997802734374999-8.707594535319272e-17j), (-0.039978027343750035-9.412454112709393e-17j), (-0.039978027343749924-6.103426170921443e-17j), (-0.03997802734374992-6.808285748311547e-17j), (-0.03997802734374993-6.593015803148929e-17j), (-0.03997802734374996-7.513145325701665e-17j), (-0.039978027343749944-6.593015803148932e-17j), (-0.03997802734374996-7.513145325701665e-17j), (-0.039978027343749965-7.297875380539039e-17j), (-0.039978027343749965-8.218004903091778e-17j), (-0.03997802734374996-6.808285748311551e-17j), (-0.039978027343749986-7.513145325701667e-17j), (-0.03997802734374997-7.297875380539043e-17j), (-0.039978027343749965-8.218004903091777e-17j), (-0.03997802734374996-7.297875380539044e-17j), (-0.039978027343749986-8.218004903091775e-17j), (-0.03997802734374998-8.00273495792915e-17j), (-0.03997802734375001-8.922864480481896e-17j), (-0.039978027343749944-6.808285748311555e-17j), (-0.03997802734374996-7.513145325701663e-17j), (-0.039978027343749944-7.297875380539037e-17j), (-0.03997802734374999-8.218004903091781e-17j), (-0.03997802734374995-7.513145325701662e-17j), (-0.03997802734374997-8.218004903091773e-17j), (-0.03997802734375001-8.00273495792916e-17j), (-0.03997802734375003-8.707594535319277e-17j), (-0.03997802734374997-7.297875380539044e-17j), (-0.03997802734374998-8.00273495792916e-17j), (-0.03997802734375-8.00273495792915e-17j), (-0.03997802734375003-8.707594535319279e-17j), (-0.03997802734375-8.218004903091784e-17j), (-0.03997802734375001-8.922864480481898e-17j), (-0.03997802734375002-8.922864480481892e-17j), (-0.039978027343750035-9.627724057872012e-17j), (-0.03997802734374998-6.808285748311561e-17j), (-0.03997802734374996-7.513145325701666e-17j), (-0.039978027343749986-7.513145325701665e-17j), (-0.03997802734375-8.00273495792916e-17j), (-0.03997802734374997-7.513145325701665e-17j), (-0.03997802734374997-8.218004903091782e-17j), (-0.03997802734374998-8.00273495792915e-17j), (-0.03997802734375002-8.707594535319279e-17j), (-0.03997802734374998-7.513145325701668e-17j), (-0.039978027343749986-8.218004903091781e-17j), (-0.03997802734375-8.002734957929162e-17j), (-0.039978027343750014-8.922864480481897e-17j), (-0.03997802734375-8.218004903091781e-17j), (-0.039978027343750056-8.707594535319278e-17j), (-0.039978027343750056-8.707594535319279e-17j), (-0.03997802734375004-9.627724057872014e-17j), (-0.039978027343749986-7.513145325701672e-17j), (-0.03997802734374999-8.218004903091781e-17j), (-0.03997802734374997-8.218004903091778e-17j), (-0.039978027343750014-8.922864480481897e-17j), (-0.03997802734375-8.218004903091783e-17j), (-0.03997802734375001-8.922864480481895e-17j), (-0.03997802734374999-8.707594535319265e-17j), (-0.03997802734375002-9.627724057872011e-17j), (-0.039978027343750035-8.00273495792917e-17j), (-0.03997802734375003-8.707594535319277e-17j), (-0.03997802734375001-8.922864480481901e-17j), (-0.03997802734375004-9.627724057872014e-17j), (-0.03997802734375003-8.707594535319277e-17j), (-0.03997802734375004-9.627724057872012e-17j), (-0.039978027343750035-9.627724057872012e-17j), (-0.039978027343750076-1.0332583635262138e-16j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j])