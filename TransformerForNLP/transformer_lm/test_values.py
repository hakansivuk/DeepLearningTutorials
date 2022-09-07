#!/usr/bin/env python
# coding=utf-8


# ******************************************************************************
# DO NOT MODIFY THIS FILE
# ANY MODIFICATION WILL LEAD TO FAILING THE HOMEWORK
# ******************************************************************************

from dataclasses import dataclass
import torch


@dataclass
class SelfAttentionTestValues:
    x = torch.tensor([[[ 1.7155067921, -0.5655270815,  0.6405809522, -0.1572896242,
          -1.1750909090, -0.1397917271,  0.4590217769],
         [-0.0533850193,  1.6443881989,  0.7343022823,  0.5214210749,
           0.6633939743, -0.4956606925, -1.3677116632],
         [ 1.3678129911,  1.6993682384, -0.4744907618,  2.6529147625,
          -1.2602540255,  2.1358072758, -1.5857182741],
         [-0.5047985315, -1.8141431808, -0.4335966408, -0.4154182971,
           0.1066478193,  1.7869369984,  0.6697169542],
         [ 1.0141265392, -0.7674757838, -0.7085445523,  0.6600707173,
          -0.3026537597,  0.3295344114, -1.5546015501]],

        [[-0.5166931748,  0.7564309835, -1.0381063223,  1.0672849417,
          -0.1894607991,  1.2309306860, -1.1008110046],
         [ 1.6145222187,  1.3697460890, -0.5526681542, -0.7901086211,
          -1.2283296585, -1.0574961901, -0.6041626930],
         [-0.6468274593, -0.0077255247, -0.4526945949,  1.1265181303,
           1.6799982786,  0.5840811729, -0.6979223490],
         [-0.4738301933,  0.2204809189, -0.6866548061, -0.9081042409,
          -1.2692421675,  0.6100322604, -0.3341373503],
         [ 2.2720429897, -0.6394405961,  0.3040486574,  0.3988451958,
          -1.4756067991,  0.0364458337, -0.0675973818]],

        [[ 0.6211856008, -0.7236230373, -1.6301867962, -1.4174182415,
          -0.3135940135, -0.0136082545, -0.8329831362],
         [ 0.0957541615,  0.8538393378, -0.6086826921,  0.7061074972,
          -3.0071625710,  0.7315099239, -0.4143024385],
         [-0.0199358203,  0.6206738949, -0.3043945134, -0.0978155956,
          -0.1981846988,  0.0538761020, -1.4279841185],
         [ 1.4950748682,  0.0146861766, -0.1261405051, -0.3552599847,
          -1.0404813290, -0.4770316184,  1.8749661446],
         [ 0.0682953075,  2.4809682369,  0.6320472956,  0.5085248947,
          -0.1852406561,  0.2135295719,  0.9339079261]]])

    k_weight = torch.tensor([[ 3.3753e-01,  5.0974e-02,  7.9800e-02, -2.9945e-01, -1.8077e-01, -5.2550e-02, -1.8518e-01],
        [ 5.4843e-02,  2.7282e-01,  2.3855e-01, -1.2515e-01, -2.0701e-01, -2.0289e-01,  2.5263e-01],
        [-3.4491e-01, -2.1460e-01, -2.8392e-01, -9.2632e-02,  1.7362e-01, -3.1761e-01, -5.2054e-02],
        [ 1.5732e-01, -3.5705e-01,  1.4585e-01, -5.1598e-02, -2.1706e-02, -3.2081e-01, -1.8253e-01],
        [ 5.3187e-02, -3.0583e-01,  3.0642e-01, -4.9601e-02,  3.1532e-01, 1.0194e-02, -3.6183e-01],
        [-3.4086e-01,  1.0191e-01, -3.2957e-01,  2.2999e-01, -3.3385e-01, -3.3410e-01, -2.2540e-02],
        [ 3.7520e-01,  7.9024e-02,  8.5625e-02,  3.4990e-01,  2.5223e-01, -2.0514e-04,  1.4083e-01],
        [ 2.6043e-01, -3.0859e-01, -2.4823e-01, -1.7921e-01, -2.1897e-01, 6.4627e-02, -2.0889e-01],
        [-5.1262e-03, -1.3523e-01, -2.7373e-01, -1.3004e-01, -2.3244e-01, -8.9701e-02,  2.5886e-01]])

    k_bias = torch.tensor([-0.1029, -0.3076, -0.0415, -0.2236, -0.2601,  0.0791, -0.1111, -0.0842, -0.2843])

    q_weight = torch.tensor([[ 0.3472, -0.0515,  0.0561,  0.2730,  0.0711, -0.3009,  0.1470],
            [-0.1847, -0.0879,  0.1661,  0.0717, -0.2267, -0.0344,  0.0649],
            [-0.0834, -0.3695, -0.2577,  0.2176, -0.3288,  0.1688, -0.2930],
            [ 0.0186,  0.3333,  0.1078,  0.0429,  0.0213, -0.2440, -0.2289],
            [-0.1293,  0.1577,  0.2292,  0.0689, -0.1560, -0.2222,  0.0194],
            [ 0.3286, -0.1493,  0.2783, -0.0432, -0.3487,  0.2109,  0.1952],
            [-0.3062,  0.1288,  0.2780,  0.2956,  0.0655,  0.1234, -0.2211],
            [ 0.3619, -0.0544,  0.3770,  0.0227, -0.1900, -0.0495, -0.3667],
            [ 0.1243,  0.3720, -0.2151,  0.2099, -0.0631,  0.3530,  0.0850]])

    q_bias = torch.tensor([-0.0158,  0.1086, -0.0610,  0.2846, -0.3236,  0.0257, -0.2400,  0.0742,
            0.3662])

    v_weight = torch.tensor([[ 0.1406,  0.2914, -0.1344,  0.0583, -0.2047, -0.2044, -0.1220],
            [ 0.2428,  0.3312,  0.3048, -0.2569, -0.1942,  0.1743,  0.1444],
            [-0.0720, -0.1378,  0.0879,  0.0047, -0.1575, -0.2740, -0.2612],
            [-0.0726, -0.0963, -0.1957,  0.0676, -0.1417,  0.1016, -0.1425],
            [-0.0922,  0.0423,  0.0922, -0.1990, -0.1084, -0.3636, -0.2726],
            [ 0.2086,  0.0689, -0.2987, -0.3311,  0.3007,  0.3209,  0.1314],
            [ 0.1079,  0.1849, -0.0318,  0.3577,  0.3171,  0.1348,  0.0069],
            [ 0.2667,  0.1966,  0.0442,  0.1659, -0.1463, -0.0660,  0.0752],
            [-0.2575, -0.0170, -0.3282, -0.0482,  0.1595,  0.1581, -0.0723]])

    v_bias = torch.tensor([0.0665,  0.1393,  0.3385, -0.2680, -0.1448, -0.0670,  0.3351, -0.3425, 0.2412])

    output = torch.tensor([[[ 0.2105157673,  0.2228333205,  0.3484930396, -0.0294795521,
          -0.3754127920, -0.0770054236,  0.5414814353, -0.0919139087,
           0.1169599965],
         [ 0.2016819417,  0.2830637395,  0.3614825308, -0.0635812283,
          -0.3587890267, -0.0838998929,  0.4603230655, -0.0681921616,
           0.0380719677],
         [-0.0047201216,  0.1614188403,  0.3340333402, -0.0420754664,
          -0.4140963256,  0.0388284288,  0.3270034194, -0.2439679354,
           0.1451242864],
         [ 0.1770765632,  0.2260076106,  0.2620601058, -0.0357266739,
          -0.4382422268, -0.0495859012,  0.6675907373, -0.1352395117,
           0.2252372056],
         [ 0.0480816960,  0.1435844004,  0.3212687969, -0.0292846672,
          -0.4109572470,  0.0145819094,  0.4430864453, -0.2309568226,
           0.1983086467]],

        [[ 0.4052675962,  0.2335330248,  0.3668530881, -0.0509275496,
          -0.1429520845,  0.0066072159,  0.3851401806, -0.1464658082,
           0.2766237855],
         [ 0.5521272421,  0.4795960784,  0.4395815730, -0.0762620345,
          -0.0255550891, -0.0292686112,  0.1646553725, -0.0088052303,
           0.0741291940],
         [ 0.4160813391,  0.2503903210,  0.3738204539, -0.0655715689,
          -0.1541346759, -0.0124618486,  0.4028368592, -0.1041470021,
           0.2191133350],
         [ 0.3880774379,  0.1939303130,  0.3500603139, -0.0191882588,
          -0.1456970125,  0.0298003294,  0.3861096203, -0.2051932812,
           0.3705404103],
         [ 0.5921315551,  0.4915230274,  0.4401747584, -0.0595572367,
           0.0347335860,  0.0049881861,  0.1161453351, -0.0500331372,
           0.1519682109]],

        [[ 0.5435267687,  0.4988487363,  0.3554362655, -0.0624767318,
          -0.0745642930, -0.0022981651,  0.0979103744, -0.0500706024,
           0.1773099154],
         [ 0.5071299076,  0.4655891061,  0.3597065806, -0.0611182116,
          -0.0788023770,  0.0696057528,  0.0265155062, -0.0741721690,
           0.1725510210],
         [ 0.4909020662,  0.4907126725,  0.3296431005, -0.1150105894,
          -0.0887586176,  0.0848236904,  0.0768915415, -0.0655917376,
           0.1528079361],
         [ 0.6195869446,  0.5630137324,  0.3771289289, -0.0021897629,
          -0.1000460833, -0.1458825916,  0.0717891976,  0.0318890326,
           0.1272034645],
         [ 0.5341457725,  0.6350766420,  0.2723901570, -0.1677252352,
          -0.1638271958, -0.0157902092,  0.1190564111,  0.0594849028,
           0.0320884585]]])

@dataclass
class MultiHeadAttentionTestValues:
    x = torch.tensor([[[ 1.7155067921, -0.5655270815,  0.6405809522, -0.1572896242,
          -1.1750909090, -0.1397917271,  0.4590217769],
         [-0.0533850193,  1.6443881989,  0.7343022823,  0.5214210749,
           0.6633939743, -0.4956606925, -1.3677116632],
         [ 1.3678129911,  1.6993682384, -0.4744907618,  2.6529147625,
          -1.2602540255,  2.1358072758, -1.5857182741],
         [-0.5047985315, -1.8141431808, -0.4335966408, -0.4154182971,
           0.1066478193,  1.7869369984,  0.6697169542],
         [ 1.0141265392, -0.7674757838, -0.7085445523,  0.6600707173,
          -0.3026537597,  0.3295344114, -1.5546015501]],

        [[-0.5166931748,  0.7564309835, -1.0381063223,  1.0672849417,
          -0.1894607991,  1.2309306860, -1.1008110046],
         [ 1.6145222187,  1.3697460890, -0.5526681542, -0.7901086211,
          -1.2283296585, -1.0574961901, -0.6041626930],
         [-0.6468274593, -0.0077255247, -0.4526945949,  1.1265181303,
           1.6799982786,  0.5840811729, -0.6979223490],
         [-0.4738301933,  0.2204809189, -0.6866548061, -0.9081042409,
          -1.2692421675,  0.6100322604, -0.3341373503],
         [ 2.2720429897, -0.6394405961,  0.3040486574,  0.3988451958,
          -1.4756067991,  0.0364458337, -0.0675973818]],

        [[ 0.6211856008, -0.7236230373, -1.6301867962, -1.4174182415,
          -0.3135940135, -0.0136082545, -0.8329831362],
         [ 0.0957541615,  0.8538393378, -0.6086826921,  0.7061074972,
          -3.0071625710,  0.7315099239, -0.4143024385],
         [-0.0199358203,  0.6206738949, -0.3043945134, -0.0978155956,
          -0.1981846988,  0.0538761020, -1.4279841185],
         [ 1.4950748682,  0.0146861766, -0.1261405051, -0.3552599847,
          -1.0404813290, -0.4770316184,  1.8749661446],
         [ 0.0682953075,  2.4809682369,  0.6320472956,  0.5085248947,
          -0.1852406561,  0.2135295719,  0.9339079261]]])

    k_weight = torch.tensor([[-0.1063, -0.2981, -0.2877,  0.0834, -0.0484, -0.2841,  0.0741],
            [ 0.2081,  0.1952, -0.0076,  0.3058,  0.2864,  0.1610, -0.3160],
            [-0.1172,  0.3356, -0.3691,  0.3330,  0.2582, -0.0116, -0.0570],
            [-0.0907, -0.3500,  0.0926, -0.2760,  0.2580,  0.3088,  0.0141],
            [ 0.1969,  0.0543, -0.1333,  0.1248,  0.0725, -0.2461, -0.3687],
            [ 0.1826,  0.0006,  0.2499, -0.1551,  0.3619, -0.3272, -0.1125],
            [-0.2507,  0.0077,  0.3133, -0.0574,  0.2056,  0.3713,  0.1884],
            [ 0.3352,  0.1314,  0.3396,  0.0830,  0.0756,  0.2060, -0.2216],
            [-0.0353, -0.0902, -0.3377,  0.1158, -0.2283, -0.2176, -0.1722]])

    k_bias = torch.tensor([[-0.0977,  0.2944, -0.1985, -0.3557,  0.0340,  0.3178, -0.3472,  0.3554,
            0.0233]])

    q_weight = torch.tensor([[-0.3086, -0.3213, -0.2087, -0.3292,  0.0161,  0.1240, -0.1918],
            [ 0.3426, -0.3612, -0.0703, -0.2647, -0.2496, -0.1062, -0.3202],
            [ 0.2433, -0.1647,  0.3558, -0.0590, -0.1165,  0.2194,  0.2306],
            [-0.2464, -0.0250,  0.1321,  0.0618, -0.1357, -0.1904,  0.1829],
            [-0.2420,  0.3560, -0.2383,  0.2626,  0.3624, -0.2009, -0.3375],
            [ 0.1099, -0.0289, -0.1738,  0.1841,  0.0936, -0.0057,  0.1738],
            [ 0.1633,  0.2466, -0.1937,  0.2488, -0.1198, -0.0965, -0.0947],
            [-0.2140,  0.0946,  0.0468, -0.3024, -0.1060, -0.2296,  0.3220],
            [-0.0914,  0.1320,  0.3296, -0.1026,  0.0033, -0.0633,  0.3722]])

    q_bias = torch.tensor([[ 0.2806, -0.3478, -0.1584,  0.0882,  0.1134, -0.0092,  0.3748, -0.1630,
            -0.2767]])

    v_weight = torch.tensor([[ 0.3592,  0.2064,  0.2418, -0.1996,  0.1366, -0.0531, -0.2488],
            [ 0.2508,  0.2152, -0.3537, -0.2907,  0.1745, -0.2720, -0.2226],
            [ 0.2919, -0.0619,  0.2787, -0.0545,  0.2280, -0.1651,  0.1713],
            [-0.0683,  0.1587,  0.0817,  0.3362, -0.2900,  0.1133, -0.1054],
            [-0.2760, -0.2162, -0.0619, -0.2918,  0.3263,  0.0891, -0.1367],
            [-0.2570,  0.1090, -0.3126,  0.2004, -0.2389, -0.3132,  0.0722],
            [-0.3225, -0.0174,  0.0016, -0.1845,  0.3386,  0.0697,  0.1470],
            [ 0.2257, -0.0421, -0.0932, -0.1081, -0.2401, -0.3732, -0.3712],
            [ 0.1299, -0.0834,  0.2622,  0.2889, -0.3740,  0.0352, -0.0522]])

    v_bias = torch.tensor([0.0698,  0.2133, -0.3360, -0.1818,  0.1197,  0.3382,  0.0172,  0.0095,
            0.2389])

    mix_weight = torch.tensor([[-0.0124,  0.1029, -0.1511, -0.2357,  0.2790, -0.0330, -0.2161, -0.2450,
            0.3125],
            [ 0.0711, -0.2523,  0.2517, -0.0668, -0.0294, -0.0599,  0.3029,  0.1537,
            0.1411],
            [-0.1946,  0.1453,  0.0961,  0.0325, -0.0906,  0.2282, -0.2104, -0.0410,
            0.0766],
            [ 0.0024,  0.0054,  0.1708,  0.2593,  0.1127,  0.1083, -0.2145, -0.2557,
            0.2348],
            [ 0.0568,  0.0524, -0.2769,  0.1221, -0.1206, -0.2086, -0.0268, -0.2428,
            0.1555],
            [-0.0541,  0.1258, -0.0052, -0.1051,  0.1895, -0.1966, -0.1909, -0.3158,
            0.0093],
            [ 0.3195,  0.3082, -0.1145, -0.2525, -0.2618,  0.0785, -0.0407,  0.1298,
            0.1172],
            [-0.2913, -0.1322, -0.2259, -0.2706,  0.2585, -0.1997, -0.0789, -0.1245,
            -0.1720],
            [ 0.0957, -0.2951,  0.0657,  0.0746,  0.0132,  0.0820, -0.0872, -0.3193,
            -0.1184]])

    mix_bias = torch.tensor([-0.1311, -0.1704, -0.1757, -0.0358,  0.2548, -0.3090,  0.1808,  0.0880,
            0.2132])

    output = torch.tensor([[[ 0.2653949857, -0.3898952603, -0.1847366095,  0.0557978414,
           0.5826772451, -0.2446260154,  0.5381613374,  0.0349829048,
           0.0993775949],
         [-0.1239079461, -0.2678142786, -0.0384911895,  0.2047659904,
           0.5270562172, -0.4301512539,  0.3028242588, -0.2351409495,
           0.2518207133],
         [-0.1716510057, -0.2790189981, -0.1734310538,  0.1265947521,
           0.6342402697, -0.3591130078,  0.2729218006, -0.1754840910,
           0.3829123080],
         [ 0.1854580641, -0.3336180747, -0.0747076944,  0.1195029318,
           0.4872711003, -0.2949740589,  0.4431051612, -0.0147035718,
           0.0810063928],
         [-0.0793497935, -0.3325383067, -0.1365037709,  0.1134977043,
           0.6219005585, -0.3726818562,  0.3974629045, -0.1818723083,
           0.2547298670]],

        [[-0.0316435024, -0.4254347086, -0.0447596014, -0.0172012113,
           0.4211370051, -0.3986284137,  0.6170557737, -0.1705637872,
           0.0650518090],
         [ 0.0124605596, -0.4472484291, -0.0374664664, -0.0024553798,
           0.4301339388, -0.3821820319,  0.5673350096, -0.1118354946,
           0.0603661835],
         [-0.0165807679, -0.4340479672, -0.0211899430, -0.0159413069,
           0.4102503955, -0.4149373770,  0.6523191333, -0.1851630211,
           0.0259219706],
         [ 0.0510785133, -0.4559706450, -0.0084297061,  0.0228060633,
           0.3995627761, -0.3932098150,  0.5950405598, -0.1312637180,
           0.0082806200],
         [ 0.0420520306, -0.4754657149, -0.0367158800, -0.0333881043,
           0.4430341721, -0.3814690709,  0.4932700396,  0.0026190281,
           0.0482328385]],

        [[-0.0071125478, -0.5180098414,  0.1134183258,  0.0697242171,
           0.4594281912, -0.4477581978,  0.6845968962, -0.2616634071,
           0.0189009607],
         [ 0.0279899985, -0.4742868245,  0.0798530132,  0.0629927218,
           0.4410182238, -0.4218569696,  0.6373345852, -0.1959256530,
           0.0599437654],
         [ 0.0139437467, -0.4843077064,  0.1008827537,  0.0782286227,
           0.4152255058, -0.4332765341,  0.6252264380, -0.2161866426,
           0.0344226062],
         [ 0.0148519129, -0.4440045059,  0.0705540925,  0.0187833607,
           0.4086564183, -0.4656288922,  0.6868534684, -0.2060250342,
           0.0149074346],
         [ 0.0535991192, -0.4203821123,  0.0749258846,  0.0517452322,
           0.3492925167, -0.4276905358,  0.5954324007, -0.1474812180,
           0.0226201266]]])

    output_causal = torch.tensor([[[ 0.0475579351, -0.0913476124,  0.0440149903,  0.1785248220,
           0.3548521996, -0.4394596517,  0.7646304965, -0.4328724742,
           0.0092312396],
         [-0.0573108047, -0.1454855651, -0.0549971759,  0.1268195808,
           0.2814910710, -0.4598471522,  0.6341501474, -0.3961671293,
           0.0585360527],
         [-0.1899628043, -0.3055714965,  0.0054942071,  0.2208038121,
           0.5284106135, -0.5802035928,  0.6590940356, -0.5854513645,
           0.0861331075],
         [ 0.1876077056, -0.2837544084, -0.0967689976,  0.1519246101,
           0.5077774525, -0.2736586332,  0.3478276134,  0.0270225033,
           0.1719466895],
         [-0.0793497935, -0.3325383067, -0.1365037709,  0.1134977043,
           0.6219005585, -0.3726818562,  0.3974629045, -0.1818723083,
           0.2547298670]],

        [[ 0.1246989667, -0.6746543646,  0.0042384714,  0.1149082035,
           0.6301747561, -0.3612523079,  0.2317223251,  0.1384659708,
           0.1769501120],
         [ 0.0192187130, -0.6343839169,  0.0919291824,  0.0175712779,
           0.4713328481, -0.4529335499,  0.8346425295, -0.3025983572,
          -0.0838809907],
         [ 0.0087695569, -0.5480879545, -0.0298912674, -0.0307853427,
           0.4220149517, -0.3416909277,  0.7868588567, -0.2475206554,
          -0.0150731802],
         [ 0.0659394711, -0.5367902517, -0.0470927060, -0.0458458848,
           0.4026753902, -0.3561277688,  0.5213494301,  0.0050901026,
           0.0173581839],
         [ 0.0420520306, -0.4754657149, -0.0367158800, -0.0333881043,
           0.4430341721, -0.3814690709,  0.4932700396,  0.0026190281,
           0.0482328385]],

        [[ 0.1791678071, -0.6072083116, -0.1195452660, -0.5024091005,
           0.0328460932, -0.2673108280,  0.8773642182,  0.2677236199,
          -0.4611311257],
         [ 0.0736232698, -0.6530120969,  0.1768705398, -0.0291518141,
           0.3941786885, -0.5521944165,  0.7183887362, -0.1328623295,
          -0.1967453361],
         [ 0.0255488008, -0.6130273938,  0.0934471041, -0.0636577010,
           0.3733022213, -0.5275705457,  0.7253204584, -0.1601575315,
          -0.1924727261],
         [ 0.0770485550, -0.5037287474,  0.0946947187, -0.0605903454,
           0.3633129597, -0.4753720760,  0.7602209449, -0.1355814040,
          -0.1266511381],
         [ 0.0535991192, -0.4203821123,  0.0749258846,  0.0517452322,
           0.3492925167, -0.4276905358,  0.5954324007, -0.1474812180,
           0.0226201266]]])
