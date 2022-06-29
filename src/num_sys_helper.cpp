#include "num_sys_helper.h"
#include <cmath>

using namespace std;

string bit_flip(string bit_arr, int bit_ind)
{
    int bit_ind_LSB = bit_arr.size() - 1 - bit_ind;
    bit_arr[bit_ind_LSB] = ((int)bit_arr[bit_ind_LSB] - '0' != 0)? '0': '1';
    return bit_arr;
}

string int_to_bin(int num)
{
    string s = "";
    while(num > 0)
    {
        int r = num % 2;
        s = to_string(r) + s;
        num /= 2;
    }
    return s;
}

float bin_to_int(string frac_str)
{
    int power_count = 0;
    float frac = 0;

    for (auto &ch : frac_str)
    {
        frac += ((int)ch - '0') * pow(2, power_count);
        power_count++;
    }
    return frac;
}

string int_to_bitstream(int num, int exp_len)
{
    string sign = (num < 0)? "1" : "0";
    num = abs(num);
    string int_str = int_to_bin(num);

    if (int_str.size() > exp_len)
    {
        int_str = string(exp_len, '1');
    }

    int_str = string(exp_len - int_str.size(), '0') + int_str;
    return int_str;
}

int bitstream_to_int(string bit_arr, int exp_len)
{
    string exp_str = bit_arr.substr(1, exp_len);
    return bin_to_int(exp_str);
}
