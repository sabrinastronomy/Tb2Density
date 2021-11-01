import stats
import matplotlib.pyplot as plt
# cmodel_train = stats.Model("cgan", "c_gen_5000.h5")
cmodel_val = stats.Model("cgan", "c_gen_5000.h5")
# cmodel.animate("c_gen_5000")

kmodes, pk = cmodel_val.twod_power_spectrum(cmodel_val.arr, 1, 20)
o_kmodes, o_pk = cmodel_val.twod_power_spectrum(cmodel_val.val_arr, 1, 20)
diff = (o_kmodes**2 * pk) - kmodes**2 * pk
plt.plot(kmodes, diff)
plt.show()
plt.close()
plt.plot(kmodes, kmodes**2 * pk, label="Sample Generated Data")
plt.plot(o_kmodes, o_kmodes**2 * pk, label="Original Generated Data", ls='--')
plt.legend()
plt.title("Rough: Generated Data Power Spectra")
plt.xlabel("k")
plt.ylabel("$k^{2}P(k)$")
plt.show()