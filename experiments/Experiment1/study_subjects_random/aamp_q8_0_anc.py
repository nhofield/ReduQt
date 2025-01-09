input_space = 'zero'

def run(qc):
    qc.initialize([(0.027102209894482657+8.449819517184238e-17j), (0.0271022098944828+8.148610105566766e-17j), (0.027102209894482695+8.148610105566698e-17j), (0.027102209894482754+7.847400693949181e-17j), (0.02710220989448266+5.778691688153805e-17j), (0.027102209894482646+5.477482276536309e-17j), (0.027102209894482664+7.847400693949151e-17j), (0.02710220989448266+7.546191282331602e-17j), (0.027102209894482678+8.148610105566717e-17j), (0.027102209894482795+7.847400693949198e-17j), (0.02710220989448275+7.847400693949175e-17j), (0.027102209894482747+7.546191282331638e-17j), (0.02710220989448281+7.847400693949193e-17j), (0.027102209894482733+7.546191282331626e-17j), (0.027102209894482747+7.546191282331662e-17j), (0.027102209894482737+7.244981870714092e-17j), (0.02710220989448274+5.778691688153832e-17j), (0.02710220989448271+7.847400693949227e-17j), (0.02710220989448275+7.84740069394915e-17j), (0.02710220989448278+7.546191282331675e-17j), (-0.6735908430097823-3.836900851402789e-15j), (0.027102209894482743+5.1762728649187124e-17j), (0.027102209894482705+5.1762728649187155e-17j), (0.027102209894482757+7.244981870714106e-17j), (0.02710220989448278+7.847400693949208e-17j), (0.027102209894482785+7.546191282331676e-17j), (0.027102209894482813+7.546191282331707e-17j), (0.027102209894482754+7.244981870714112e-17j), (0.027102209894482733+5.176272864918735e-17j), (0.02710220989448263+7.244981870714044e-17j), (0.027102209894482674+7.244981870714066e-17j), (0.027102209894482608+6.943772459096485e-17j), (0.027102209894482747+8.148610105566708e-17j), (0.027102209894482757+5.477482276536325e-17j), (0.027102209894482702+7.847400693949215e-17j), (0.027102209894482747+7.546191282331617e-17j), (0.027102209894482858+5.4774822765362203e-17j), (-0.6735908430097823-4.318254865666247e-15j), (0.027102209894482643+7.546191282331606e-17j), (0.027102209894482563+4.8750634533011064e-17j), (0.027102209894482764+7.847400693949202e-17j), (0.02710220989448284+7.546191282331653e-17j), (0.02710220989448273+7.546191282331651e-17j), (0.027102209894482705+7.244981870714081e-17j), (0.02710220989448269+7.546191282331654e-17j), (0.02710220989448271+4.875063453301211e-17j), (0.0271022098944827+7.244981870714084e-17j), (0.027102209894482674+6.943772459096494e-17j), (0.027102209894482733+7.847400693949192e-17j), (0.02710220989448271+7.546191282331656e-17j), (0.027102209894482768+7.54619128233163e-17j), (0.02710220989448271+7.244981870714054e-17j), (0.027102209894482625+5.176272864918692e-17j), (0.02710220989448283+4.875063453301186e-17j), (0.027102209894482636+7.244981870714026e-17j), (0.02710220989448265+6.943772459096491e-17j), (0.027102209894482667+7.546191282331618e-17j), (0.027102209894482664+7.244981870714071e-17j), (0.027102209894482743+7.244981870714112e-17j), (0.027102209894482723+6.943772459096533e-17j), (0.0271022098944827+7.244981870714059e-17j), (0.027102209894482608+6.943772459096518e-17j), (0.027102209894482646+6.94377245909649e-17j), (0.027102209894482664+6.642563047478964e-17j), (0.027102209894482667+8.14861010556672e-17j), (0.027102209894482733+7.847400693949176e-17j), (0.02710220989448271+7.847400693949187e-17j), (0.027102209894482757+7.546191282331645e-17j), (0.027102209894482775+7.847400693949198e-17j), (0.02710220989448265+7.546191282331614e-17j), (0.02710220989448271+7.546191282331624e-17j), (0.027102209894482775+7.244981870714108e-17j), (0.027102209894482674+7.847400693949169e-17j), (0.027102209894482688+7.546191282331633e-17j), (0.027102209894482664+7.546191282331617e-17j), (0.027102209894482705+7.244981870714075e-17j), (0.027102209894482605+7.546191282331574e-17j), (0.02710220989448266+7.244981870714102e-17j), (0.027102209894482657+7.244981870714066e-17j), (0.0271022098944827+6.94377245909652e-17j), (0.027102209894482695+7.847400693949161e-17j), (0.027102209894482674+7.546191282331635e-17j), (0.027102209894482733+7.546191282331655e-17j), (0.027102209894482737+7.24498187071409e-17j), (0.027102209894482643+5.17627286491875e-17j), (0.02710220989448269+7.244981870714123e-17j), (0.027102209894482678+7.244981870714065e-17j), (0.027102209894482664+6.943772459096502e-17j), (0.02710220989448264+7.546191282331619e-17j), (0.027102209894482705+7.244981870714073e-17j), (0.02710220989448272+7.244981870714103e-17j), (0.02710220989448273+6.943772459096538e-17j), (0.02710220989448261+7.244981870714034e-17j), (0.027102209894482775+6.943772459096554e-17j), (0.02710220989448271+6.943772459096527e-17j), (0.027102209894482657+6.642563047478959e-17j), (0.027102209894482688+7.847400693949185e-17j), (0.027102209894482705+7.546191282331644e-17j), (0.02710220989448275+7.546191282331654e-17j), (0.027102209894482754+7.244981870714079e-17j), (0.027102209894482643+7.54619128233161e-17j), (0.027102209894482754+4.875063453301065e-17j), (0.02710220989448268+7.244981870714027e-17j), (0.027102209894482664+6.943772459096486e-17j), (0.02710220989448272+7.546191282331628e-17j), (0.02710220989448272+7.244981870714097e-17j), (0.027102209894482712+7.244981870714074e-17j), (0.027102209894482716+6.943772459096545e-17j), (0.02710220989448269+7.24498187071408e-17j), (0.027102209894482695+6.943772459096522e-17j), (0.027102209894482674+6.943772459096536e-17j), (0.027102209894482747+6.642563047478993e-17j), (0.02710220989448273+7.546191282331616e-17j), (0.027102209894482684+7.244981870714076e-17j), (0.027102209894482743+7.244981870714077e-17j), (0.027102209894482712+6.943772459096523e-17j), (0.02710220989448267+7.24498187071394e-17j), (0.02710220989448267+6.943772459096476e-17j), (0.0271022098944827+6.943772459096501e-17j), (0.027102209894482667+6.642563047478959e-17j), (0.02710220989448265+7.244981870714049e-17j), (0.02710220989448268+6.943772459096522e-17j), (0.02710220989448272+6.94377245909652e-17j), (0.02710220989448269+6.642563047478956e-17j), (0.027102209894482695+6.943772459096516e-17j), (0.027102209894482688+6.642563047478943e-17j), (0.027102209894482726+6.642563047478995e-17j), (0.02710220989448267+6.341353635861393e-17j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j])