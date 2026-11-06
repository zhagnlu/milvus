
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>
#include "common/GallopingSearch.h"
using namespace milvus;
using PkType = std::variant<int64_t,std::string>;
using Timestamp=uint64_t;
struct SegOffset { int64_t n; explicit SegOffset(int64_t n):n(n){} };

void bench(size_t n,size_t m,size_t chunks,bool dense){
 std::vector<int64_t> rows(n);for(size_t i=0;i<n;++i)rows[i]=i*2;
 std::mt19937_64 rng(53901);
 std::vector<PkType> pks(m);for(size_t i=0;i<m;++i)pks[i]=int64_t(dense ? n+2*i : rng()%(2*n));
 std::sort(pks.begin(),pks.end());
 uint64_t total=0;
 auto get_timestamp=[](size_t){return Timestamp(1);};
 auto read_ts=[](int64_t){return Timestamp(0);};
 auto timestamp_hit=[](Timestamp a,Timestamp b){return a<b;};
 auto callback=[&](SegOffset o,Timestamp t){total+=o.n+t;};
 const size_t repeat=std::max(size_t(2),std::min(size_t(300),size_t(10000)/m));
 auto run=[&](bool merge){
  auto begin=std::chrono::steady_clock::now();
  for(size_t r=0;r<repeat;++r){
    size_t first_pk = 0;
    auto search_sorted_chunk = [&](size_t rows,
                                   int64_t base,
                                   auto value_at,
                                   auto key_at) {
        if (rows == 0) {
            return;
        }
        // Segment PKs are globally sorted. Keep groups that can also occur in
        // the next chunk: advancing to the end of this chunk's matches would
        // lose duplicate PKs straddling a chunk boundary.
        first_pk =
            GallopingLowerBound(pks.size(), first_pk, value_at(0), key_at);
        size_t pos = 0;
        for (size_t j = first_pk;
             j < pks.size() && !(value_at(rows - 1) < key_at(j));) {
            const auto target = key_at(j);
            size_t next = j + 1;
            while (next < pks.size() && key_at(next) == target) {
                ++next;
            }
            pos = GallopingLowerBound(rows, pos, target, value_at);
            size_t end = pos;
            while (end < rows && value_at(end) == target) {
                ++end;
            }
            // Search each distinct PK once, but retain every delete event and
            // its order. Different insert versions can need different deletes.
            for (size_t k = j; k < next; ++k) {
                const auto timestamp = get_timestamp(k);
                for (size_t offset = pos; offset < end; ++offset) {
                    const auto segment_offset = base + offset;
                    if (timestamp_hit(read_ts(segment_offset), timestamp)) {
                        callback(SegOffset(segment_offset), timestamp);
                    }
                }
            }
            pos = end;
            j = next;
        }
    };

   for(size_t c=0;c<chunks;++c){
    size_t first=c*n/chunks,last=(c+1)*n/chunks;
    if(merge){
     search_sorted_chunk(last-first,first,[&](size_t i){return rows[first+i];},[&](size_t i){return std::get<int64_t>(pks[i]);});
    }else{
     for(auto& pk:pks){
      auto k=std::get<int64_t>(pk);
      auto it=std::lower_bound(rows.begin()+first,rows.begin()+last,k);
      for(;it!=rows.begin()+last && *it==k;++it)callback(SegOffset(it-rows.begin()),1);
     }
    }
   }
  }
  return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/repeat;
 };
 double old=0,now=0;
 for(int r=0;r<4;++r){if(r%2){now+=run(true);old+=run(false);}else{old+=run(false);now+=run(true);}}
 printf("rows=%zu batch=%zu chunks=%zu %s binary_us=%.2f merge_us=%.2f speedup=%.2f checksum=%llu\n",n,m,chunks,dense?"dense":"random",old/4,now/4,old/now,(unsigned long long)total);
}
int main(){for(size_t n:{2000000,44000000})for(size_t chunks:{1,8})for(bool dense:{false,true})for(size_t m:{1,8,128,4096,100000})bench(n,m,chunks,dense);}
