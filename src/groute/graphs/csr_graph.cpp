// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#ifdef HAVE_METIS
#include <metis.h>
#endif

#include <unordered_set>
#include <groute/graphs/csr_graph.h>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>

DECLARE_int32(block_size);

namespace groute {
namespace graphs {

    namespace multi
    {
        MetisPartitioner::MetisPartitioner(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting METIS partitioning\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            // Convert to 64-bit for metis, idx_t is defined in metis.h
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, calling METIS\n", (int)IDXTYPEWIDTH);
            
            int result = METIS_PartGraphKway(
                &nnodes,                      // 
                &ncons,                       //
                row_start.data(),     //
                edge_dst.data(),      //
                NULL,                         // vwgt
                NULL,                         // vsize
                m_origin_graph.edge_weights ? edge_weights.data() : nullptr,  // adjwgt
                &nparts,                      // nparts
                NULL,                         // tpwgts
                NULL,                         // ubvec
                NULL,                         // options
                &edgecut,                     // objval
                &partition_table[0]);         // part [out]

            if (result != METIS_OK) {
                printf(
                    "METIS partitioning failed (%s error), Exiting.\n", 
                    result == METIS_ERROR_INPUT ? "input" : result == METIS_ERROR_MEMORY ? "memory" : "general");
                exit(0);
            }

            printf("Building partitioned graph and lookup tables\n");

            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);

            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;
                }

                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("METIS partitioning done\n");
#endif
        }

        void MetisPartitioner::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> MetisPartitioner::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

        std::vector<index_t> GetUniqueHalos(
            const index_t* edge_dst,
            index_t seg_snode, index_t seg_nnodes,
            index_t seg_sedge, index_t seg_nedges, int& halos_counter)
        {
            std::unordered_set<index_t> halos_set;
            halos_counter = 0;

            for (int i = 0; i < seg_nedges; ++i)
            {
                index_t dest = edge_dst[seg_sedge + i];
                if (dest < seg_snode || dest >= (seg_snode + seg_nnodes)) // an halo
                {
                    ++halos_counter;
                    halos_set.insert(dest);
                }
            }

            std::vector<index_t> halos_vec(halos_set.size());
            std::copy(halos_set.begin(), halos_set.end(), halos_vec.begin());

            return halos_vec;
        }

/* ---------------------------- Naive Partitioner --------------------------- */

        NaivePartitioner::NaivePartitioner(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting naive partitioning\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            // Convert to 64-bit for metis
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, doing naive partitioning\n", (int)IDXTYPEWIDTH);
            
            // int result = METIS_PartGraphKway(
            //     &nnodes,                      // 
            //     &ncons,                       //
            //     row_start.data(),     //
            //     edge_dst.data(),      //
            //     NULL,                         //
            //     NULL,                         //
            //     m_origin_graph.edge_weights ? edge_weights.data() : nullptr,  //
            //     &nparts,                      //
            //     NULL,                         //
            //     NULL,                         //
            //     NULL,                         //
            //     &edgecut,                     //
            //     &partition_table[0]);         //

            // FQ: we only need to modify partition_table
            idx_t nnodes_per_seg = nnodes / nparts;
            idx_t _t = 0;
            for (idx_t i = 0; i < nnodes; i++)
            {
                _t = i / nnodes_per_seg;
                if (_t >= nparts-1)
                    partition_table[i] = nparts-1;
                else
                    partition_table[i] = i / nnodes_per_seg;
                // printf("%d", partition_table[i]);
            }

            for (idx_t i = 0; i < 10; i++)
                printf("%d", partition_table[i]);

            printf("Building partitioned graph and lookup tables\n");

            // FQ: This struct store the relation between node ID and it partition ID
            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);    // FQ: allocate 1 such struct for every node

            // FQ: init the data
            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            // FQ: sort, put nodes belong to one partition together
            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            // FQ: reorganize partitioned graph in CSR
            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;    // FQ: record the boundary node, the first node in seg
                }

                // FQ: construct a lookup table between old and new node ID, because we sort the node_partitions
                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;       

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("Naive partitioning done\n");
#endif
        }

        void NaivePartitioner::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> NaivePartitioner::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

/* ---------------- Metis Partitioner with weight assignment ---------------- */

        MetisPartitionerDegreeW::MetisPartitionerDegreeW(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting METIS partitioning with vertex weights (degree)\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            printf("0..");
            fflush(stdout);
            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            printf("1..");
            fflush(stdout);

            // Convert to 64-bit for metis
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            printf("2..");fflush(stdout);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            printf("3..");fflush(stdout);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, computing degrees\n", (int)IDXTYPEWIDTH);

            idx_t *vdegrees = (idx_t*) malloc(sizeof(idx_t) * nnodes);
            for (idx_t i = 0; i < nnodes; i++)
            {
                vdegrees[i] = m_origin_graph.row_start[i+1] - m_origin_graph.row_start[i];
            }

            printf("Degree computed, calling METIS\n"); 
            
            int result = METIS_PartGraphKway(
                &nnodes,                      // 
                &ncons,                       //
                row_start.data(),     //
                edge_dst.data(),      //
                vdegrees,                         // vwgt
                NULL,                         // vsize
                m_origin_graph.edge_weights ? edge_weights.data() : nullptr,  // adjwgt
                &nparts,                      // nparts
                NULL,                         // tpwgts
                NULL,                         // ubvec
                NULL,                         // options
                &edgecut,                     // objval
                &partition_table[0]);         // part [out]

            if (result != METIS_OK) {
                printf(
                    "METIS partitioning failed (%s error), Exiting.\n", 
                    result == METIS_ERROR_INPUT ? "input" : result == METIS_ERROR_MEMORY ? "memory" : "general");
                exit(0);
            }
            // free(vdegrees);

            printf("Building partitioned graph and lookup tables\n");

            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);

            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;
                }

                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("METIS partitioning done\n");
#endif
        }

        void MetisPartitionerDegreeW::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> MetisPartitionerDegreeW::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

/* -------------------------- TB Graph Constructor -------------------------- */

        TBGraphConstructor::TBGraphConstructor(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting TB Graph construction...\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            // Convert to 64-bit for metis
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, doing TB Graph construction...\n", (int)IDXTYPEWIDTH);

            clock_t start = clock();
            const idx_t TB_SIZE = 1024;
            idx_t TB_NUM = std::ceil((float)nnodes / TB_SIZE);    // total number of TBs
            // new vectors to store TB graph
            std::vector<idx_t> TB_row_start(TB_NUM+1);
            std::vector<idx_t> TB_edge_dst;
            // weights of TB Graph
            std::vector<idx_t> TB_v_wgt(TB_NUM);
            std::vector<idx_t> TB_e_wgt;
            idx_t new_TB_edge_num = 0;
            idx_t nE = 0;

            // for each future TB, scan all its vertex (of original graph)
            for (idx_t tbid = 0; tbid < TB_NUM; tbid++)
            {
                if (tbid % 10 == 0)
                {
                    printf("Processing TB #%lld...edge #%lld\r", tbid, nE);
                    fflush(stdout);
                }
                idx_t srcTB = tbid;
                idx_t current_edge_num = 0;

                // for each vertex of current TB
                for (idx_t vid = tbid * TB_SIZE; vid < (tbid+1) * TB_SIZE; vid++)
                {
                    if (vid >= nnodes)  // could only be true at the last TB
                        break;

                    // for each edge of current vertex
                    idx_t start = row_start[vid];
                    idx_t end = row_start[vid+1];
                    for (idx_t e = start; e < end; e++, nE++)
                    {
                        idx_t dstTB = edge_dst[e] / TB_SIZE;

                        if (srcTB == dstTB)
                        {
                            // intra-TB edge, w[TB]++
                            TB_v_wgt[srcTB]++;
                        }
                        else
                        {
                            // inter-TB edge, add edge if not existed, w[edge]++

                            // check if dstTB has been inserted into TB_edge_dst
                            idx_t TB_target_e = -1;
                            std::vector<idx_t>::iterator TB_e_start = TB_edge_dst.begin() + TB_row_start[srcTB];
                            std::vector<idx_t>::iterator TB_e_end = TB_e_start + current_edge_num;
                            std::vector<idx_t>::iterator found = std::lower_bound(TB_e_start, TB_e_end, dstTB);
                            if (found != TB_e_end && *found == dstTB)    // found
                                TB_target_e = std::distance(TB_edge_dst.begin(), found);
                            

                            // if found
                            if (TB_target_e != -1)
                            {
                                TB_e_wgt[TB_target_e]++;
                            }
                            else
                            {
                                // if not found, insert new TB edge
                                current_edge_num++;
                                TB_target_e = std::distance(TB_edge_dst.begin(), found);
                                TB_edge_dst.insert(TB_edge_dst.begin()+TB_target_e, dstTB);
                                TB_e_wgt.insert(TB_e_wgt.begin()+TB_target_e, (idx_t)1);
                            }
                            
                        }   // end srcTB != dstTB
                        
                    }   // end for-loop of e
                }   // end for-loop of vid

                // update TB index
                TB_row_start[srcTB+1] = TB_row_start[srcTB] + current_edge_num;
            }   // end for-loop of tbid
            clock_t end = clock();
            printf("TB Graph has been constructed in %.3f seconds. It has %d TBs and %d connections.\n", (float)(end-start)/CLOCKS_PER_SEC, (int)TB_NUM, (int)TB_edge_dst.size());

/* ----------------------------------- CDF ---------------------------------- */
#define CDF
#ifdef CDF
            // CDF: export the degree of each TB
            printf("Writing to cdf.txt...\n");
            std::ofstream fd;
            fd.open("/data/qfan005/cdf.txt", std::ios::out);    // this file will be read with numpy.fromfile() so directly output the array separated with a whitespace
            // fd << TB_NUM << std::endl;
            for (idx_t i = 0; i < TB_NUM; i++)
            {
                fd << TB_row_start[i+1] - TB_row_start[i] << " ";
            }
            fd.close();
#endif  
/* --------------------- Adjacency matrix visualization --------------------- */
#define ADJMAT
#ifdef ADJMAT
            // AdjMat figure
            // Will use scipy.sparse.csr_graph to parse the file, so just output (indptr, indices, data)
            // the first line: <# TBs> <# conns>
            printf("Writing to adjmat.txt...\n");
            fd.open("/data/qfan005/adjmat.txt", std::ios::out);
            fd << TB_NUM << " " << TB_edge_dst.size() << std::endl;
            // the second line: indptr a.k.a. TB_row_start
            for (idx_t i = 0; i < TB_NUM+1; i++)
                fd << TB_row_start[i] << " ";
            fd << std::endl;
            // the second line: indices a.k.a. TB_edge_dst
            for (std::vector<idx_t>::iterator it = TB_edge_dst.begin(); it < TB_edge_dst.end(); it++)
                fd << *it << " ";
            fd << std::endl;
            // the third line: data a.k.a. TB_e_wgt
            for (std::vector<idx_t>::iterator it = TB_e_wgt.begin(); it < TB_e_wgt.end(); it++)
                fd << *it << " ";
            fd << std::endl;
            fd.close();
#endif

            // exit(0);

/* --------------------------- Partition TB Graph --------------------------- */

            printf("Partitioning TB Graph with METIS recursive bisection...\n#TB=%lld, #TB=%lld, #wTB=%lld, #Con=%lld, #wCon=%lld\n",
                TB_row_start.size(), TB_NUM, TB_v_wgt.size(), TB_edge_dst.size(), TB_e_wgt.size());
            fflush(stdout);
            std::vector<idx_t> partition_table_TB(TB_NUM);
            // // increase vertex id by 1, because vertex id starts from 1 rather than 0
            
            start = clock();
            int result = METIS_PartGraphRecursive(
                &TB_NUM,    // nvtxs
                &ncons,    // ncon, number of weights associated with v
                TB_row_start.data(),    // xadj
                TB_edge_dst.data(),     // adjncy
                NULL,        // vwgt // TODO: w/ or w/o ?
                NULL,                   // vsize
                TB_e_wgt.data(),        // adjwgt
                &nparts,                // nparts
                NULL,
                NULL,
                NULL,
                &edgecut,               // edgecut
                &partition_table_TB[0]
            );
            end = clock();
            printf("Done, time=%.3f seconds, edgecut=%lld (undirected)\n", (float)(end-start)/CLOCKS_PER_SEC, edgecut);
            fflush(stdout);

            printf("Constructing partition table for data graph...\n");
            fflush(stdout);
            for (idx_t i = 0; i < nnodes; i++)
            {
                if (i % 10000 == 0)
                {
                    printf("%lld\r", i);
                    fflush(stdout);
                }
                idx_t TBid = i / TB_SIZE;
                partition_table[i] = partition_table_TB[TBid];
            }

            // printf("Calculating # of vertices of each partition...\n");
            // std::vector<idx_t> partitions_TB_v(8);
            // for (idx_t i = 0; i < nnodes; i++)
            // {
            //     partitions_TB_v[partition_table[i]]++;
            // }
            // for (idx_t i = 0; i < 8; i++)
            //     printf("%lld ", partitions_TB_v[i]);
            // printf("\n");

            // printf("Calculating # of TBs of each partition...\n");
            // std::vector<idx_t> partitions_TB(8);
            // for (idx_t i = 0; i < TB_NUM; i++)
            // {
            //     partitions_TB[partition_table_TB[i]]++;
            // }
            // for (idx_t i = 0; i < 8; i++)
            //     printf("%lld ", partitions_TB[i]);
            // printf("\n");
            // exit(0);

            // printf("Calculating edgecut...\n");
            // fflush(stdout);
            // edgecut = 0;
            // for (idx_t i = 0; i < nnodes; i++)
            // {
            //     if (i % 10000 == 0)
            //     {
            //         printf("%lld\r", i);
            //         fflush(stdout);
            //     }
                    
            //     for (idx_t j = row_start[i]; j < row_start[i+1]; j++)
            //     {
            //         if (partition_table[i] != partition_table[edge_dst[j]])
            //             edgecut++;
            //     }
            // }
            // printf("edgecut=%lld\n", edgecut);
            // fflush(stdout);

            // Comparison: partition data graph directly with METIS k-way
            // printf("Partitioning data graph with METIS k-way...");
            // fflush(stdout);
            // start = clock();
            // result = METIS_PartGraphKway(
            //     &nnodes,                      // 
            //     &ncons,                       //
            //     row_start.data(),     //
            //     edge_dst.data(),      //
            //     NULL,                         //
            //     NULL,                         //
            //     nullptr,                      // adjwgt
            //     &nparts,                      //
            //     NULL,                         //
            //     NULL,                         //
            //     NULL,                         //
            //     &edgecut,                     //
            //     &partition_table[0]);         //
            // end = clock();
            // printf("Done, time=%.3f seconds, edgecut=%lld (undirected)\n", (float)(end-start)/CLOCKS_PER_SEC, edgecut);
            // fflush(stdout);

            // printf("Calculating edgecut...\n");
            // fflush(stdout);
            // edgecut = 0;
            // for (idx_t i = 0; i < nnodes; i++)
            // {
            //     if (i % 10000 == 0)
            //     {
            //         printf("%lld\r", i);
            //         fflush(stdout);
            //     }
                    
            //     for (idx_t j = row_start[i]; j < row_start[i+1]; j++)
            //     {
            //         if (partition_table[i] != partition_table[edge_dst[j]])
            //             edgecut++;
            //     }
            // }
            // printf("edgecut=%lld\n", edgecut);
            // fflush(stdout);

            // printf("Calculating # of vertices of each partition...\n");
            // std::vector<idx_t> partitions_v(8);
            // for (idx_t i = 0; i < nnodes; i++)
            // {
            //     partitions_v[partition_table[i]]++;
            // }
            // for (idx_t i = 0; i < 8; i++)
            //     printf("%lld ", partitions_v[i]);
            // printf("\n");

            // exit(0);

            printf("Building partitioned graph and lookup tables\n");

            // FQ: This struct store the relation between node ID and it partition ID
            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);    // FQ: allocate 1 such struct for every node

            // FQ: init the data
            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            // FQ: sort, put nodes belong to one partition together
            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            // FQ: reorganize partitioned graph in CSR
            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;    // FQ: record the boundary node, the first node in seg
                }

                // FQ: construct a lookup table between old and new node ID, because we sort the node_partitions
                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;       

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("Naive partitioning done\n");
#endif
        }

        void TBGraphConstructor::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> TBGraphConstructor::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

/* ------------------------- Real Random Partitioner ------------------------ */

        RealRandomPartitioner::RealRandomPartitioner(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting Real Random Partitioning...\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            // Convert to 64-bit for metis
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, doing Real Random Partitioning...\n", (int)IDXTYPEWIDTH);

            clock_t start = clock();
            srand(2333);
            for (idx_t i = 0; i < nnodes; i++)
                partition_table[i] = rand() % nparts;
            clock_t end = clock();
            printf("Real random partitioned in %.3f seconds.\n", (float)(end-start)/CLOCKS_PER_SEC);


            // printf("Calculating # of vertices of each partition...\n");
            // std::vector<idx_t> partitions_v(nparts);
            // for (idx_t i = 0; i < nnodes; i++)
            // {
            //     partitions_v[partition_table[i]]++;
            // }
            // for (idx_t i = 0; i < nparts; i++)
            //     printf("%lld ", partitions_v[i]);
            // printf("\n");

            // printf("Calculating edgecut...\n");
            // fflush(stdout);
            // edgecut = 0;
            // for (idx_t i = 0; i < nnodes; i++)
            // {
            //     if (i % 10000 == 0)
            //     {
            //         printf("%lld\r", i);
            //         fflush(stdout);
            //     }
                    
            //     for (idx_t j = row_start[i]; j < row_start[i+1]; j++)
            //     {
            //         if (partition_table[i] != partition_table[edge_dst[j]])
            //             edgecut++;
            //     }
            // }
            // printf("edgecut=%lld\n", edgecut/2);
            // fflush(stdout);

            // exit(0);

            printf("Building partitioned graph and lookup tables\n");

            // FQ: This struct store the relation between node ID and it partition ID
            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);    // FQ: allocate 1 such struct for every node

            // FQ: init the data
            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            // FQ: sort, put nodes belong to one partition together
            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            // FQ: reorganize partitioned graph in CSR
            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;    // FQ: record the boundary node, the first node in seg
                }

                // FQ: construct a lookup table between old and new node ID, because we sort the node_partitions
                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;       

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("Naive partitioning done\n");
#endif
        }

        void RealRandomPartitioner::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> RealRandomPartitioner::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }
    
/* ------ METIS Partitioner with edge weight assignment (max endpoints) ----- */

        MetisPartitionerEW_MaxV::MetisPartitionerEW_MaxV(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting METIS partitioning with vertex weights (degree)\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            printf("0..");
            fflush(stdout);
            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            printf("1..");
            fflush(stdout);

            // Convert to 64-bit for metis
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            printf("2..");fflush(stdout);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            printf("3..");fflush(stdout);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, computing edge weights (max endpoints)\n", (int)IDXTYPEWIDTH);fflush(stdout);

            // v degrees
            std::vector<idx_t> vdegrees(nnodes);
            for (idx_t i = 0; i < nnodes; i++)
            {
                vdegrees[i] = m_origin_graph.row_start[i+1] - m_origin_graph.row_start[i];
            }
            printf("Degree computed, computing edge weights\n"); fflush(stdout);

            // e wgt
            std::vector<idx_t> e_wgt(nedges);
            for (idx_t v = 0; v < nnodes; v++)
            {
                for (idx_t e = row_start[v]; e < row_start[v+1]; e++)
                {
                    e_wgt[e] = std::max(vdegrees[v], vdegrees[edge_dst[e]]);
                }
            }

            printf("Degree computed, calling METIS\n"); fflush(stdout);
            
            int result = METIS_PartGraphKway(
                &nnodes,                      // 
                &ncons,                       //
                row_start.data(),     //
                edge_dst.data(),      //
                NULL,                         // vwgt
                NULL,                         // vsize
                e_wgt.data(),  // adjwgt
                &nparts,                      // nparts
                NULL,                         // tpwgts
                NULL,                         // ubvec
                NULL,                         // options
                &edgecut,                     // objval
                &partition_table[0]);         // part [out]

            if (result != METIS_OK) {
                printf(
                    "METIS partitioning failed (%s error), Exiting.\n", 
                    result == METIS_ERROR_INPUT ? "input" : result == METIS_ERROR_MEMORY ? "memory" : "general");
                exit(0);
            }

            printf("Building partitioned graph and lookup tables\n");

            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);

            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;
                }

                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("METIS partitioning with edge weights (max) done\n");
#endif
        }

        void MetisPartitionerEW_MaxV::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> MetisPartitionerEW_MaxV::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

/* ------ METIS Partitioner with edge weight assignment (LCC) ----- */

        MetisPartitionerEW_LCC::MetisPartitionerEW_LCC(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting METIS partitioning with edge weights (LCC)\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            printf("0..");
            fflush(stdout);
            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            printf("1..");
            fflush(stdout);

            // Convert to 64-bit for metis
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            printf("2..");fflush(stdout);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            printf("3..");fflush(stdout);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, computing edge weights (LCC)\n", (int)IDXTYPEWIDTH);fflush(stdout);

            std::vector<idx_t> e_wgt(nedges, 0);

            std::string inputfile = FLAGS_graphfile + ".wgt";
            std::ifstream fin;
            fin.open(inputfile, std::ios::in);
            if (!fin)
            {
                std::vector<double> LCC(nnodes, 0);
                for (idx_t v = 0; v < nnodes; v++)
                {
                    if (v % 10000 == 0)
                    {
                        printf("%lld\r", v);
                        fflush(stdout);
                    }
                    std::vector<idx_t>::iterator start = edge_dst.begin() + row_start[v];
                    std::vector<idx_t>::iterator end = edge_dst.begin() + row_start[v+1];
                    for (idx_t u = row_start[v]; u < row_start[v+1]; u++)
                    {
                        // for u's neighbors (v's neighbor's neighbors), find in v's neighbors
                        for (idx_t x = row_start[edge_dst[u]]; x < row_start[edge_dst[u]+1]; x++)
                        {
                            std::vector<idx_t>::iterator found = std::lower_bound(start, end, edge_dst[x]);
                            if (found != end && *found == edge_dst[x])
                            {
                                LCC[v]++;
                            }
                        }
                    }
                    idx_t k = row_start[v+1] - row_start[v];
                    LCC[v] = 2 * LCC[v] / (k * (k-1));
                }
                printf("LCC computed\n");fflush(stdout);

                for (idx_t v = 0; v < nnodes; v++)
                {
                    for (idx_t u = row_start[v]; u < row_start[v+1]; u++)
                    {
                        e_wgt[u] += (idx_t)(LCC[v] * 100000);

                        // update (u,v)
                        std::vector<idx_t>::iterator start = edge_dst.begin() + row_start[edge_dst[u]];
                        std::vector<idx_t>::iterator end = edge_dst.begin() + row_start[edge_dst[u]+1];
                        std::vector<idx_t>::iterator found = std::lower_bound(start, end, v);
                        if (found != end && *found == v)
                        {
                            idx_t distance = std::distance(edge_dst.begin(), found);
                            e_wgt[distance] += (idx_t)(LCC[v] * 100000);
                        }
                    }
                }

                // check zeros
                idx_t zeros = 0;
                for (idx_t e = 0; e < nedges; e++)
                {
                    if (e_wgt[e] <= 0)
                    {
                        e_wgt[e] = 1;
                        zeros++;
                    }
                }

                // write to file
                std::string outputfile = FLAGS_graphfile + ".wgt";
                std::ofstream fout;
                fout.open(outputfile, std::ios::out);
                for (idx_t e = 0; e < nedges; e++)
                    fout << e_wgt[e] << " ";
                fout.close();
                printf("Edge weights computed, %lld zeros, calling METIS\n", zeros); fflush(stdout);
            }   // if (!fin)
            else
            {
                // read from file
                for (idx_t e = 0; e < nedges; e++)
                    fin >> e_wgt[e];
                printf("Edge weights read, calling METIS\n"); fflush(stdout);
                fin.close();
            }
            
            
            int result = METIS_PartGraphKway(
                &nnodes,                      // 
                &ncons,                       //
                row_start.data(),     //
                edge_dst.data(),      //
                NULL,                         // vwgt
                NULL,                         // vsize
                e_wgt.data(),  // adjwgt
                &nparts,                      // nparts
                NULL,                         // tpwgts
                NULL,                         // ubvec
                NULL,                         // options
                &edgecut,                     // objval
                &partition_table[0]);         // part [out]

            if (result != METIS_OK) {
                printf(
                    "METIS partitioning failed (%s error), Exiting.\n", 
                    result == METIS_ERROR_INPUT ? "input" : result == METIS_ERROR_MEMORY ? "memory" : "general");
                exit(0);
            }
            // free(vdegrees);

            printf("Building partitioned graph and lookup tables\n");

            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);

            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;
                }

                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("METIS partitioning with edge weights (LCC) done\n");
#endif
        }

        void MetisPartitionerEW_LCC::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> MetisPartitionerEW_LCC::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

/* ------------------ Locality-Aware TB Graph partitioning ------------------ */

        LocalityAwareTBPartitioner::LocalityAwareTBPartitioner(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting Locality-Aware TB Graph partitioning\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            // Convert to 64-bit for metis, idx_t is defined in metis.h
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, calling METIS\n", (int)IDXTYPEWIDTH);
            
            // calculate #TBs
            idx_t nTBs = nnodes / FLAGS_block_size + 1;
            idx_t nTBs_per_GPU = nTBs / nparts + 1;
            if (nTBs_per_GPU == 0)
                nTBs_per_GPU = 1;
            
            int result = METIS_PartGraphRecursive(
                &nnodes,                      // 
                &ncons,                       //
                row_start.data(),     //
                edge_dst.data(),      //
                NULL,                         // vwgt
                NULL,                         // vsize
                m_origin_graph.edge_weights ? edge_weights.data() : nullptr,  // adjwgt
                &nTBs,                      // nparts
                NULL,                         // tpwgts
                NULL,                         // ubvec
                NULL,                         // options
                &edgecut,                     // objval
                &partition_table[0]);         // part [out]

            // TODO: re-order before mapping to GPU

            // group TBs into GPUs
            for (idx_t i = 0; i < nnodes; i++)
            {
                partition_table[i] = partition_table[i] / nTBs_per_GPU;
            }

            if (result != METIS_OK) {
                printf(
                    "METIS partitioning failed (%s error), Exiting.\n", 
                    result == METIS_ERROR_INPUT ? "input" : result == METIS_ERROR_MEMORY ? "memory" : "general");
                exit(0);
            }

            printf("Building partitioned graph and lookup tables\n");

            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);

            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;
                }

                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("METIS partitioning done\n");
#endif
        }

        void LocalityAwareTBPartitioner::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> LocalityAwareTBPartitioner::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

    }   // namespace multi
}
}
