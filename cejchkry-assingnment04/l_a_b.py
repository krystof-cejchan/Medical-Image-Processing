from utils import get_avg_lab_vector

filename = './files/Sample_color.bmp'

avg_vector, lab_data = get_avg_lab_vector(filename)

print(f"\tavg L\t{avg_vector[0]:.2f}")
print(f"\tavg a\t{avg_vector[1]:.2f}")
print(f"\tavg b\t{avg_vector[2]:.2f}")
print(avg_vector)